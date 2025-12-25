from ultralytics import YOLO
from pathlib import Path
import subprocess
import shutil
import torch


class YoloExporter:
    def __init__(self, prefix):
        self.__prefix = prefix
        self.__yolov8 = {
            "yolov8n",
            "yolov8s",
            "yolov8m",
            "yolov8l",
            "yolov8x",
            "yolo11n",
            "yolo11s",
            "yolo11m",
            "yolo11l",
            "yolo11x",
        }

        self.__yolov5 = {"yolov5s", "yolov5m", "yolov5l", "yolov5x"}

        BASE_DIR = Path(__file__).resolve().parents[2]
        self.__out_dir = BASE_DIR / "data" / "engine_models"
        self.__out_dir.mkdir(parents=True, exist_ok=True)

    def export_to_trt(
        self,
        model_path: str,
        model_name: str,
        img_size: int = 640,
        batch: int = 8,
        fp16: bool = True,
    ) -> str:
        assert torch.cuda.is_available(), "CUDA is required"

        model_path = Path(model_path).resolve
        name = Path(model_path).name.split(".")[0]

        if model_name in self.__yolov8:
            target = self.__out_dir / f"{self.__prefix }_{name}.engine"
            if target.exists():
                print(f"YOLOv8 TensorRT engine already exists at: {target}")
                return target

            if model_path.suffix != ".pt":
                raise ValueError("YOLOv8 export requires .pt weights")

            model = YOLO(str(model_path))
            model.export(
                format="engine",
                imgsz=img_size,
                batch=batch,
                half=fp16,
                device=0,
                workspace=4,
                dynamic=False,
            )

            generated = model_path.with_suffix(".engine")
            generated.replace(target)

            print(f"YOLOv8/11 TensorRT engine saved at: {target}")
            return target

        elif model_name in self.__yolov5:
            engine_path = self.__out_dir / f"{self.__prefix }_{name}.engine"
            if engine_path.exists():
                print(f"YOLOv5 TensorRT engine already exists at: {engine_path}")
                return engine_path

            if model_path.suffix == ".pt":
                if shutil.which("yolov5") is None:
                    raise RuntimeError("yolov5 CLI not found. pip install yolov5")

                onnx_path = self.__out_dir / f"{self.__prefix }_{name}.onnx"
                if not onnx_path.exists():
                    subprocess.run(
                        [
                            "yolov5",
                            "export",
                            "--weights",
                            str(model_path),
                            "--img",
                            str(img_size),
                            "--batch",
                            str(batch),
                            "--include",
                            "onnx",
                        ],
                        check=True,
                    )

                model_path = onnx_path

            if model_path.suffix != ".onnx":
                raise ValueError("YOLOv5 requires .pt or .onnx")

            trtexec_cmd = [
                "trtexec",
                f"--onnx={model_path}",
                f"--saveEngine={engine_path}",
                "--explicitBatch",
                f"--minShapes=images:1x3x{img_size}x{img_size}",
                f"--optShapes=images:{batch}x3x{img_size}x{img_size}",
                f"--maxShapes=images:{batch}x3x{img_size}x{img_size}",
            ]

            if fp16:
                trtexec_cmd.append("--fp16")

            subprocess.run(trtexec_cmd, check=True)

            print(f"YOLOv5 TensorRT engine saved at: {engine_path}")
            return engine_path

        else:
            raise ValueError(f"Unsupported model: {model_name}")
