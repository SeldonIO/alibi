from alibi.api.types import Array
from pydantic import BaseModel, conlist, confloat
from typing import Any, List, Optional, Union
from typing_extensions import Literal

import numpy as np

ModelType = Literal['blackbox', 'whitebox']
ExpType = Literal['local', 'global']


class AlibiBaseModel(BaseModel):
    class Config:
        json_encoders = {
            np.ndarray: lambda x: numpy_encoder(x)
        }


class DefaultMeta(AlibiBaseModel):
    name: Optional[str] = None
    type: List[ModelType] = []
    explanations: List[ExpType] = []
    params: dict = {}


# Anchors

## Common

class AnchorDataRawTabularExamples(AlibiBaseModel):
    covered_true: Array[Any, (-1, -1)]
    covered_false: Array[Any, (-1, -1)]
    uncovered_true: Array[Any, (-1, -1)]
    uncovered_false: Array[Any, (-1, -1)]


class AnchorDataRawImageExamples(AlibiBaseModel):
    covered_true: Union[List[Array], Array]
    covered_false: Union[List[Array], Array]
    uncovered_true: Union[List[Array], Array]
    uncovered_false: Union[List[Array], Array]


class AnchorDataRawTextExamples(AlibiBaseModel):
    covered_true: Array[str, (-1,)]
    covered_false: Array[str, (-1,)]
    uncovered_true: Array[str, (-1,)]
    uncovered_false: Array[str, (-1,)]


class AnchorDataRawTabular(AlibiBaseModel):
    feature: List[int]
    mean: List[float]
    precision: List[confloat(ge=0.0, le=1.0)]
    coverage: List[confloat(ge=0.0, le=1.0)]
    examples: conlist(AnchorDataRawTabularExamples, min_items=2, max_items=2)
    all_precision: float
    num_preds: int
    success: bool
    names: List[str]
    prediction: int
    instance: Array[Any, (-1,)]
    instances: Array[Any, (-1, 1)]


class AnchorDataRawImage(AlibiBaseModel):
    feature: List[int]
    mean: List[float]
    precision: List[confloat(ge=0.0, le=1.0)]
    coverage: List[confloat(ge=0.0, le=1.0)]
    examples: List[AnchorDataRawImageExamples]
    all_precision: float
    num_preds: int
    success: bool
    prediction: int
    instance: Array
    instances: Array


class AnchorDataRawText(AlibiBaseModel):
    feature: List[int]
    mean: List[float]
    precision: List[confloat(ge=0.0, le=1.0)]
    coverage: List[confloat(ge=0.0, le=1.0)]
    examples: conlist(AnchorDataRawTextExamples, min_items=1, max_items=1)
    all_precision: float
    num_preds: int
    success: bool
    names: List[str]
    prediction: int
    instance: str
    instances: List[str]


class AnchorDataTabular(AlibiBaseModel):
    anchor: List[str]
    precision: confloat(ge=0.0, le=1.0)
    coverage: confloat(ge=0.0, le=1.0)
    raw: AnchorDataRawTabular


class AnchorDataImage(AlibiBaseModel):
    anchor: Array
    precision: confloat(ge=0.0, le=1.0)
    coverage: confloat(ge=0.0, le=1.0)
    raw: AnchorDataRawImage
    segments: Array


class AnchorDataText(AlibiBaseModel):
    anchor: List[str]
    precision: confloat(ge=0.0, le=1.0)
    coverage: confloat(ge=0.0, le=1.0)
    raw: AnchorDataRawText


class ExplanationModel(AlibiBaseModel):
    meta: DefaultMeta
    data: Union[AnchorDataTabular, AnchorDataText, AnchorDataImage]


def numpy_encoder(obj: Any) -> Any:
    if isinstance(
            obj,
            (
                    np.int_,
                    np.intc,
                    np.intp,
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.uint8,
                    np.uint16,
                    np.uint32,
                    np.uint64,
            ),
    ):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
