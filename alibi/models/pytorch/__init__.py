from alibi.utils.missing_optional_dependency import import_optional

ADULTEncoder, ADULTDecoder, MNISTEncoder, MNISTDecoder, MNISTClassifier = import_optional(
    'alibi.models.pytorch.cfrl_models',
    names=['ADULTEncoder', 'ADULTDecoder', 'MNISTEncoder', 'MNISTDecoder', 'MNISTClassifier'])

HeAE, AE = import_optional('alibi.models.pytorch.autoencoder', names=['HeAE', 'AE'])
Actor, Critic = import_optional('alibi.models.pytorch.actor_critic', names=['Actor', 'Critic'])
