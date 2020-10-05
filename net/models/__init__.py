
from .build import MODEL_REGISTRY, build_model # noqa
# print("MODEL_REGISTRY.__dict__ in init py1 ",MODEL_REGISTRY.__dict__)
from .Image_model_builder import Generator,Discriminator # noqa
# print("MODEL_REGISTRY.__dict__ in init py2 ",MODEL_REGISTRY.__dict__)

if __name__=="__main__":
    print("model init here")