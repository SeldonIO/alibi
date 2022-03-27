from alibi.utils.missing_optional_dependency import import_optional

ADULTEncoder, ADULTDecoder, MNISTEncoder, MNISTDecoder, MNISTClassifier = import_optional(
    'alibi.models.pytorch.cfrl_models',
    names=['ADULTEncoder', 'ADULTDecoder', 'MNISTEncoder', 'MNISTDecoder', 'MNISTClassifier'])

HeAE = import_optional('alibi.models.pytorch.autoencoder', names=['HeAE'])
Actor, Critic = import_optional('alibi.models.pytorch.actor_critic', names=['Actor', 'Critic'])
