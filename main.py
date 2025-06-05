import os 
import argparse
import torch
from tqdm import tqdm
from scan_processor import ScanProcessor
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_obj_path", type=str, required=True)
    parser.add_argument("--albedo_path", type=str, default=None)
    parser.add_argument("--smpl_obj_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--ortho_views", type=str, default="0,45,90,180,270,315")
    parser.add_argument("--render_res", type=int, default=1024)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parser.parse_args()
    ortho_views = [int(x) for x in args.ortho_views.split(',')]
    scan_processor = ScanProcessor(device=device, ortho_views=ortho_views)
    output_dict = scan_processor.forward(
        mesh_obj_path=args.mesh_obj_path,
        albedo_path=args.albedo_path,
        smpl_obj_path=args.smpl_obj_path,
        render_res=args.render_res
    )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        front_view_image_raw = output_dict["front_view_image_raw"]
        front_view_image_processed = output_dict["front_view_image_processed"]
        ortho_views_images = output_dict["ortho_views_images"]

        mesh = output_dict["mesh"]
        smpl_mesh = output_dict["smpl_mesh"]

        front_view_image_raw.save(os.path.join(args.output_dir, "front_view_image_raw.png"))
        front_view_image_processed.save(os.path.join(args.output_dir, "front_view_image_processed.png"))
        for i, ortho_view_image in enumerate(ortho_views_images):
            ortho_view_image.save(os.path.join(args.output_dir, f"ortho_view_{i}.png"))

        mesh.write_obj(os.path.join(args.output_dir, "mesh.obj"))
        smpl_mesh.write_obj(os.path.join(args.output_dir, "smpl_mesh.obj"))