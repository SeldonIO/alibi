from alibi.utils.missing_optional_dependency import import_optional

DistributedExplainer = import_optional('alibi.utils.distributed', names=['DistributedExplainer'])
LanguageModel = import_optional('alibi.utils.lang_model', names=['LanguageModel'])
