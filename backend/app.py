"""PathSnap API and narrowly allowlisted same-origin evaluator UI."""
from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from .boundary import PublicBoundary
from .config import Settings
from .datasets import DatasetService
from .errors import ServiceError
from .service import InferenceRuntime
from .static import public_asset


class MatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    trip_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def create_app(settings: Settings | None = None, *, runtime: InferenceRuntime | None = None,
               dataset_service: DatasetService | None = None) -> FastAPI:
    settings = settings or (runtime.settings if runtime is not None else Settings.from_env())
    runtime = runtime or InferenceRuntime(settings)
    service = dataset_service or DatasetService(runtime)

    @asynccontextmanager
    async def lifespan(app):
        # Loading is startup work, not module-import side effects or a web server.
        service.start()
        yield
        service.close()

    app = FastAPI(title="PathSnap", version="1.0.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None,
                  redirect_slashes=False)
    app.state.runtime = runtime
    app.state.settings = settings
    app.state.datasets = service

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        headers = {"Retry-After": "3"} if exc.status_code == 429 else None
        return JSONResponse({"detail": {**exc.public(), "request_id": request.state.request_id}}, status_code=exc.status_code, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Never serialize Pydantic's input/body/context fields to a public client.
        return JSONResponse({"detail": {"code": "invalid_request", "message": "Supply exactly one registered trip_id string as a JSON object.", "request_id": request.state.request_id}}, status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        messages = {404: "This resource is not public.", 405: "This HTTP method is not allowed."}
        return JSONResponse({"detail": {"code": f"http_{exc.status_code}", "message": messages.get(exc.status_code, "The request could not be served."), "request_id": request.state.request_id}}, status_code=exc.status_code)

    @app.get("/health")
    def health():
        # Primary legacy status; dataset summaries appear only once an explicit
        # evaluator partition is configured, so unconfigured output is unchanged.
        return service.global_health()

    @app.get("/trips")
    def trips():
        return runtime.trips()

    @app.post("/match")
    def match(body: MatchRequest, request: Request):
        return runtime.match(body.trip_id, request.state.request_id)

    @app.get("/datasets")
    def datasets():
        return service.catalogue()

    @app.get("/datasets/{dataset_id}/trips")
    def dataset_trips(dataset_id: str):
        return service.trips(dataset_id)

    @app.post("/datasets/{dataset_id}/match")
    def dataset_match(dataset_id: str, body: MatchRequest, request: Request):
        return service.match(dataset_id, body.trip_id, request.state.request_id)

    # This catch-all is deliberately registered AFTER API endpoints. It is not a
    # general-purpose directory mount and has no listing, fallback or data route.
    @app.api_route("/{asset_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    def frontend(asset_path: str, request: Request):
        asset = public_asset(settings.frontend_dir, asset_path)
        if asset is None:
            raise StarletteHTTPException(404)
        target, media_type = asset
        return FileResponse(target, media_type=media_type)

    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_credentials=False, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"], expose_headers=["X-Request-ID"], max_age=600)
    # Outermost user middleware gives preflight/error/static responses the same
    # request IDs, no-store policy and anti-sniffing headers as inference replies.
    app.add_middleware(PublicBoundary, allowed_origins=settings.cors_origins)
    return app
