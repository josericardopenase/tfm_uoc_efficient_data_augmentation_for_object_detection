#!/usr/bin/env python3
"""
Converter COCO Dataset Format to YOLO Format (Ultralytics)
 
Este script convierte anotaciones en formato COCO JSON a formato YOLO txt
compatible con Ultralytics YOLOv8.
 
Estructura esperada COCO:
    dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── annotations/
        ├── instances_train.json
        ├── instances_val.json
        └── instances_test.json
 
Estructura generada YOLO:
    yolo_dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml
"""
 
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import shutil
import argparse
 
 
class COCOtoYOLOConverter:
    def __init__(self, coco_dir: str, output_dir: str = "yolo_dataset"):
        """
        Inicializa el conversor.
        
        Args:
            coco_dir: Ruta al directorio con dataset COCO
            output_dir: Ruta donde se guardará el dataset YOLO
        """
        self.coco_dir = Path(coco_dir)
        self.output_dir = Path(output_dir)
        self.coco_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def convert_bbox_coco_to_yolo(self, 
                                   bbox: List[float], 
                                   img_width: int, 
                                   img_height: int) -> Tuple[float, float, float, float]:
        """
        Convierte bounding box de formato COCO a YOLO.
        
        COCO: [x_min, y_min, width, height] (esquina superior izquierda)
        YOLO: [x_center, y_center, width, height] (normalizado 0-1)
        
        Args:
            bbox: Bounding box en formato COCO [x, y, w, h]
            img_width: Ancho de la imagen
            img_height: Alto de la imagen
            
        Returns:
            Tuple con [x_center, y_center, width, height] normalizados
        """
        x_min, y_min, width, height = bbox
        
        # Centro de la caja
        x_center = (x_min + width / 2) / img_width
        y_center = (y_min + height / 2) / img_height
        
        # Normalizar ancho y alto
        width_norm = width / img_width
        height_norm = height / img_height
        
        return x_center, y_center, width_norm, height_norm
    
    def load_coco_annotation(self, json_path: str) -> Dict:
        """Carga el archivo JSON de anotaciones COCO."""
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def process_split(self, 
                     split_name: str,
                     json_path: str,
                     images_dir: str) -> None:
        """
        Procesa un split del dataset (train, val, test).
        
        Args:
            split_name: Nombre del split (train, val, test)
            json_path: Ruta al archivo JSON de anotaciones
            images_dir: Ruta al directorio de imágenes
        """
        if not Path(json_path).exists():
            print(f"⚠️  Archivo no encontrado: {json_path}")
            return
        
        print(f"\n📂 Procesando split: {split_name}")
        
        # Cargar anotaciones COCO
        coco_data = self.load_coco_annotation(json_path)
        
        # Crear diccionarios para mapeo rápido
        images_info = {img['id']: img for img in coco_data['images']}
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Crear directorios de salida
        labels_dir = self.output_dir / "labels" / split_name
        images_output_dir = self.output_dir / "images" / split_name
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Procesar cada anotación
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # Copiar imágenes y crear archivos de etiquetas
        processed_count = 0
        for img_id, img_info in images_info.items():
            img_filename = img_info['filename']
            img_path = Path(images_dir) / img_filename
            
            if not img_path.exists():
                print(f"  ⚠️  Imagen no encontrada: {img_path}")
                continue
            
            # Copiar imagen
            output_img_path = images_output_dir / img_filename
            shutil.copy2(img_path, output_img_path)
            
            # Crear archivo de etiquetas
            img_width = img_info['width']
            img_height = img_info['height']
            
            label_filename = Path(img_filename).stem + '.txt'
            label_path = labels_dir / label_filename
            
            # Escribir anotaciones YOLO
            with open(label_path, 'w') as f:
                if img_id in annotations_by_image:
                    for ann in annotations_by_image[img_id]:
                        category_id = ann['category_id']
                        bbox = ann['bbox']
                        
                        # Convertir bbox
                        x_center, y_center, width, height = self.convert_bbox_coco_to_yolo(
                            bbox, img_width, img_height
                        )
                        
                        # Escribir en formato YOLO: <class_id> <x_center> <y_center> <width> <height>
                        f.write(f"{category_id - 1} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            processed_count += 1
        
        print(f"  ✅ Procesadas {processed_count} imágenes")
        return categories
    
    def create_yaml_config(self, 
                          categories: Dict[int, str],
                          splits: List[str] = None) -> None:
        """
        Crea el archivo data.yaml para Ultralytics.
        
        Args:
            categories: Diccionario de categorías {id: nombre}
            splits: Lista de splits procesados
        """
        if splits is None:
            splits = ['train', 'val', 'test']
        
        # Construir rutas relativas
        yaml_content = f"""# Dataset YOLO - Convertido de COCO
path: {self.output_dir.absolute()}
train: images/train
val: images/val
test: images/test
 
nc: {len(categories)}
names: {{{', '.join([f'{i}: {name}' for i, name in sorted(categories.items())])}}}
"""
        
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✅ Archivo data.yaml creado: {yaml_path}")
    
    def convert(self, 
                annotations_dir: str = "annotations",
                images_base_dir: str = "images") -> None:
        """
        Ejecuta la conversión completa.
        
        Args:
            annotations_dir: Nombre del directorio con JSONs COCO
            images_base_dir: Nombre del directorio base con imágenes
        """
        print("🚀 Iniciando conversión COCO → YOLO\n")
        
        annotations_path = self.coco_dir / annotations_dir
        images_path = self.coco_dir / images_base_dir
        
        all_categories = {}
        splits_processed = []
        
        # Procesar cada split
        for split in ['train', 'val', 'test']:
            json_file = annotations_path / f"instances_{split}.json"
            images_dir = images_path / split
            
            if json_file.exists() and images_dir.exists():
                categories = self.process_split(split, str(json_file), str(images_dir))
                if categories:
                    all_categories.update(categories)
                splits_processed.append(split)
        
        # Crear archivo YAML
        if all_categories:
            self.create_yaml_config(all_categories, splits_processed)
            print(f"\n✅ Conversión completada exitosamente!")
            print(f"📁 Dataset YOLO guardado en: {self.output_dir.absolute()}")
            print(f"📊 Categorías encontradas: {len(all_categories)}")
            print(f"📦 Splits procesados: {', '.join(splits_processed)}")
        else:
            print("\n❌ No se encontraron anotaciones para procesar")
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Convierte dataset COCO a formato YOLO (Ultralytics)"
    )
    parser.add_argument(
        "--coco-dir",
        type=str,
        default=".",
        help="Ruta al directorio raíz del dataset COCO"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="yolo_dataset",
        help="Ruta de salida para el dataset YOLO"
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default="annotations",
        help="Nombre del directorio con anotaciones JSON"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="Nombre del directorio base con imágenes"
    )
    
    args = parser.parse_args()
    
    converter = COCOtoYOLOConverter(args.coco_dir, args.output_dir)
    converter.convert(args.annotations_dir, args.images_dir)
 
 
if __name__ == "__main__":
    main()