# Accuracy improvement: shared handoff

Workers: direct messages to child UUIDs currently fail; read this file and the common-module API. No worker should start actual fits until the Lead records `training_authorized: true` in the protocol. Initial scripts/tests may be completed and handed off without fits. User authorized gateway-codex/gpt-6.1-sol with max effort. Three MAX workers own features, LR/GNB, and KNN respectively.

## Validation contract

Use only TRAIN data for tuning. Split all 80 TRAIN IDs deterministically by SHA-256 of `20261118:<trip_id>`; first 16 are outer confirmation for this new iteration, remaining 64 for three whole-trip GroupKFold splits. This holdout was not historically unseen: previous releases trained on all 80. Do not claim a new official test set. All paired new baselines are refitted using the same 64-train/fold partitions. No student or evaluator input/reference/result read by tuning workers. Freeze all feature/model/decoder decisions before one outer/student comparison. Evaluator is already scored and never a new blind test.

## Common API being implemented by Lead

`load_train_dataset()` returns the **64 tuning trips only**, from byte-hash-validated immutable old TRAIN cache. `get_folds(dataset)` gives three dicts with train/validation observations, rows and IDs.

`run_cv_trial(family, run_name, parameters, profile, output, estimator_factory, fold_ids=None, bank_budget=60000, fit_policy="weighted") -> dict`. Factory signature `(parameters, seed, n_jobs) -> estimator`. X is already transformed statelessly under profile `v1`, `context`, `compact`, or `physics`; fold-local scaling/PCA/etc lives in the estimator pipeline. `fit_policy="knn_bank"` applies old stratified weighted bank and no fit weights; otherwise pipeline last-step sample weights or bare estimator sample weights. Baseline KNN uses bank60k and fixed fold seed `20261118+7919*fold`. CPU cap two threads per worker.

Use <=6 prespecified configurations per family: fold0 screen then promote top two for full3CV (fresh full run, uniquely named outputs). All metrics and failed/worse trials remain recorded. Common persists numerical trial record and private OOF probabilities/fold models under isolated accuracy private dirs, never published. Class metrics use unchanged proxy target, offset and candidate order, including binary accuracy/F1/logloss and observation top1/equal-trip top1; per-observation edge accuracy is not binary candidate accuracy. Strict probabilities finite [0,1], both class columns sum to1 within1e-10; 1e-7 clipping retained for baseline-compatible metrics, raw logits only separate diagnostic.

## Dense OOF routing STARTED

LeadfrozeHGBcandidate+v1controls and8menu in decoder/candidate_manifest.json withseparate execution_authorization.json; original PROTOCOLunchanged. TwoCPUworkersroutefold0(27,66),fold1(37,98) whileBaggingworkerstillruns; fold2queued untilworkerfrees<=3concurrent. DoNOTedit mapmatch/accuracy_matcher.py,scripts/accuracy_decoder.py,scripts/accuracy_route_validation.py orsourcemanifestinputs. Newartifactroleworkeronlynew accuracy_artifacts.py+testsallowed. Confirmationsourcenotyetfrozen canstillfixguards, noouterdataused.

## Decoder FINAL SOURCEFREEZE Lead ownsnext

OOFdecoderhandoffreceived. Lead delegatedseparatenonOOFrolemoduleaccuracy_artifacts.py toanotherMAXchild; originaldecoderworkerdoNOTtouchconfirmationrolesoraccuracy_matcher.pyfurther. LeadneedsstartdenseOOFnow, sooriginalworkerendaftersendingfinalhashes. No furthereditsplease evenstyle.

## Decoder handoff received / finishroleonly

27syntheticdecodertests+8route/confirmationguardspassedLead. No needfurtherbroadregression. WorkerpleasefinishHANDOFFandexplicitlystatewhetherCONFIRMATIONroleloaderimplemented; nonecurrentlyfound. Leadcanimplementafteryourhandoffifnot. ThenendturnsoLeadcanfreezematcher/sourcesandstartactualruns. No otheredits needed.

## Decoder urgent handoff status

Leadreviewerconfirmedconfirmationenvelopeincompatibility, anticipatedsharednoteabove; pleaseseparatestrictroleload ifshortorleaveLeadroleimplementationbutfinishandreturn. Runningyour36+synthetictestsnowbeforefits. DecoderrawrunsMUSTwaitsourcefreeze,handoffcurrentlycriticalpath. No newresearch. Needmenus/sourcefrozenintheircurrentform.

## Decoder nextaction / API gate minimal

Leadwrotedensecentralrunner accuracy_route_validation.py consumingyourstrictwrapper; doesNOTmutatePROTOCOL.json becauseclassifierfitsrunandpinithash. Yourrequire_decoder_authorization mayacceptseparate authorizationfile/Mapping containing training_authorizedtrue+decoder_selectionbindingsembedded inNEW decoder/execution_authorization.json insteadofalteringoldprotocol. Leadwillgate usingthat separately. ModelselectedprovisionalHistGB_context_large untilBaggingcomplete; noouterdataaccess. Pleasefinishsource+synthetictests+confirmationroleandHANDOFF NOW, noextrawork needed.

## Decoder required confirmation role (before handoff only)

Leadconfirmationrunner scripts/accuracy_confirmation.py willrefitv1+selected on64tuningtrips, reserve16scoreonce. Itsbundlesfold_id=fold=3,training_scope="TRAIN64 fitting, reserved16 one-shot internal confirmation; never historicallyunseen", andsource_sha256frozenmanifestusesactualrelativekeys notcommon/factory aliasescurrently. Yourstrictfoldloadercurrentlyonly0/1/2; keepitstrict. PleaseprovideSEPARATE `load_accuracy_confirmation_model(path,expected_sha256,expected_train_trip_ids,expected_validation_trip_ids,expected_protocol_sha256)` or dedicatedvalidatedmetadatafitrole support withoutpretending3isanOOFfold. All80productionexportrolecanwait. AlternativeLeadwilladdnewconfirmloaderwithroleprooflater; doroleguardnowifshort. Do notchangealreadyfrozencommon/factories/protocol. Needcodehandoffsoon tobegindenseOOFrouteswhileBaggingruns.

## Decoder clarification / next gates

Decoderworkersourcefinishnowplease. Accept10mtransitiondeadbandslack andstationarypenaltiesasboundedablation(notproactivehorizonexpansion, nopromisegain). Retain8menufrozen. NeedimportableAPIload_accuracy_emission_model(path,expectedtrain/valid/modelsha...),AccuracyMapMatcher(config...),decoder_configurations/selectionplan. LeadwillperformactualdenseTRAINrouteexperimentscentrally; donotrunfits/rawroutes. Backendallmodels remainunchangedproduction. Bundlefold_id/fold/base_feature_version alreadyavailable. Pleasehand offcode+synthetic testswithoutfurtherexpansion.

## ACTUAL TRAINING STARTED / DO NOT EDIT MODEL SOURCES

Lead authorized PROTOCOL.json true after60synthetic tests andsourcepins. CentralbackgroundrunsHistGB,KNN,andGNBthenLR started. Do NOT modify mapmatch/accuracy_features.py,scripts/accuracy_common.py,scripts/accuracy_trees.py,scripts/accuracy_knn.py,scripts/accuracy_linear_bayes.py untilallfinish. Feature/testsreadinessnowdone. Decoder mayeditonlyitsownscopetests/menuenvelope. Baggingqueued untiloneCPUworkerfinishes. Allprivateoutputsisolated. Menu/foldregistrationandcontrolledresume alreadyfixedbeforefits; revieweroldlinecountsrefertoprevsource.

## Current source freeze notice

Featureworkerfinishwrite/tests thenreturnready; no source changes afterLeadlaunchesactualfits. KNN/LRGNBhandoffready and implementationpins nowincluded bycommon. Reviewfound preregfold/menu andrecoverysafetygaps;Leadfixedvalidate_registered_trial+validate_saved+preflight+safeverifiedresume BEFOREfits; alltrialsstillnotstarted. Decoderkeep inherited fallbackradius onlyforemptysets so candidate distribution matches classifiercache; candidate_retry_limit=12 fixesK, max_radius_m=1000 baselineemptyfallback unchanged. Do not hardfail valid farGPSthatoldcacheexpanded. ExtraK24/thinning deferred. Ensure densefeaturegeneration from rawfulltrack inference evenTRAINvalidationnot350-rowprobarchives. Needfixed8configmenu end-science thennofurtheredits.

## Latest Lead decision (read before running)

Leadcentralrunnerready. Featureworkerpleasewrite implementationnowsoLeadcanstartnumericalfits; sourceonlynotrunreadyiscriticalpath. KNNworkerfinishfactory/CLIandteststhenreturn, nofurtheroptimizationwait. Decoderbundle nowcontainsbothfold_idandfold,base_feature_version,currentmetadata fields; strictsignatureloader canuse. No worker-started actual fits. Lead has central runner scripts/accuracy_trees.py for ALL5families usingyourconstructors, fixedmenus,commonweights/pins. KNNworkerdoNOTduplicatefit; finalstageusesstandalonesklearnnotmaxkhelper toguaranteetieparity. Finisheachsourcehandoff/tests thenreturn; no newrunnerneeded. Common API file now exists; Lead is writing tests+protocol, then will launch measured scripts centrally. Source workers should finish constructors/tests and signal handoff rather than wait for UUID messaging. Their shared files are visible.

1. Canonical baseline profile `v1`; common also accepts alias `baseline38` solely for existing menu compatibility, normalizes metadata to v1.
2. LR/GNB worker: add module-level `make_lr_estimator(parameters, seed, n_jobs)` and `make_gnb_estimator(parameters, seed, n_jobs)` adapters to your existing make_estimator, or Lead will wrap in tree runner. No n_jobs parameter to sklearn LogisticRegression1.8 (deprecated) needed; factory receives budget only.
3. KNN max-k reuse implementation accepted for exploratory screens, but final promoted full3CV uses exact sklearn standalone fixed-k estimator/common pipeline for deployment-consistent ties. At most SIX score-configurations including baseline25: v1(k25,p2,brute), compact(k5,p2), compact(k15,p2), compact(k35,p2), physics(k15,p2), compact(k15,p1). Same60kbank. Top2+baseline full3. No9-scoremenu expansion now.
4. Decoder: fixed candidates K12/radius80 for first strict decoder-only ablation; changingK changes candidate-relative features and is a separate distribution/policy experiment, not "same model" decoder improvement. No selection by likelihood across differing candidate sets. Start at up to8fixedK configs testing beta2/5/10, classifierweight1/2, geometryweight0.25/1, careful motion/stationary uncertainty. observation_distance_m=0 means allobs regardlessinterval; keep dense scoring comparisons. K24 may be separately preregistered/retrained later ifneeded, not first pass.
5. Exact common fold bundle keys already listed below; add `fold_id` in envelope. `feature_version` v1 or pathsnap-candidate-v2; `base_feature_names` exact38; profile_names(profile) tuple. Candidate selected manifest will bind paths+hashes.
6. Original baselines usedfloat32fits whereas newrunnerhasfloat64 safe pipeline; pair NEW v1feature controls with newfeatures atidentical precision. Reportcomparisonvshistoricalseparately toavoidclaimingpurefeatureeffectiftinyfloatdifference. Alltargets unchanged.

## Common available / runner handoff

Lead has now created scripts/accuracy_common.py with the exact contract above. All workers must inspect it; no actual fitting until PROTOCOL.json training_authorized=true and Lead launch. LR/GNB factory currently has (family,parameters,config) but common calls (parameters,seed,n_jobs); please adapt family-specific module functions or wrappers accepting three. Do not mutate common. A module-level family_factory function rather than __main__ lambda makes source pins reliable. No need wait for messaging.

Decoder envelope is concretely saved by common as private/<output.name>/<run_name>/fold_<fold>.joblib, dict keys estimator, profile, feature_names, base_feature_names, feature_version, family, run_name, parameters, train_trip_ids, validation_trip_ids, training_scope, source_sha256, protocol_sha256. Transform baseline38 with profile before predict. v1 names/version remain v1. Allprobabilitiesfloat64strictsumtol1e-10. Candidate manifest selection will bind numerical trial JSON/private fold paths/source. Configured six selection routes are hash-smallest two validation IDs in each64tuningfold; route configs fixed constants, no all80 sigma/beta calibration.

## Confirmed menus and source coordination

LR/GNB worker proposed six each including exact v1 control. Approved as written in its message: LR v1 C1, context C0.1/1/10, compact C1, physics C1; GNB v1 smoothing1e-7 plus fold-local signed-log/skew/correlation-pruning0.90 compact smoothing1e-3/1e-2/1e-1, context1e-2, physics1e-2. Use no more than80inputcontextfeatures. Factory signature `(parameters, seed, n_jobs)`; common transforms feature profile before fit. Include baseline control on full3CV even if not top2 so comparisons are paired.

KNN worker: prioritize simple genuine sklearn pipelines first: v1 control25, compact5/15/35, physics15, compact distance p1 optional. Same60kbank. Optional multi-k query optimization is unnecessary for firstpass; if implemented, exact-cutoff ties must be disclosed and tested, not hidden. Same64tripdata common.

Private output root in common is `experiments/accuracy_improvement/private/<output.name>/<run_name>/`; trial JSON is `<output>/cv/<run_name>.json`. Fullrun saves every fold estimator `.joblib`, and OOF `probabilities.npz`; fold0screen arrays include onlyvalidationrows toavoidmistakingpartialOOFforfull. All menus/preregistrations must be on disk before fitting. Default2CPUthreadsperworkerhardcap; no service mutations.

## Scope

Feature worker: mapmatch/accuracy_features.py + tests/test_accuracy_features.py only.
LR/GNB worker: scripts/accuracy_linear_bayes.py + tests/test_accuracy_linear_bayes.py + experiments/accuracy_improvement/linear_bayes/ + private/linear_bayes/.
KNN worker: scripts/accuracy_knn.py + tests/test_accuracy_knn.py + experiments/accuracy_improvement/knn/ + private/knn/.
Lead: scripts/accuracy_common.py + scripts/accuracy_trees.py + common tests/protocol/reports/integration. No v1 or live changes.
