import torch
from model import PreTrainedDiffusion, FeatureAlignedDiffusion, ExpertEvaluator
from dataset import get_kather_dataloader
from diffusers import DDPMScheduler
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import load_model
from gen_utils import visualize_img, generate_img, retrieve_db_images
import argparse
import yaml
from dataset import CLASS_MAP


def load_yaml(yaml_path):
    with open(yaml_path) as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg


def train(
    model,
    train_dataloader,
    noise_scheduler,
    cfg,
    device
):
    print("Starting training...")
    writer = SummaryWriter(cfg["train_args"]["log"])
    # Loss and optimizers
    num_epochs = cfg["train_args"]["epochs"]
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train_args"]["lr"])
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    best_loss = 10000.0

    model.train()
    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        epoch_losses = []
        for img, label in tqdm(train_dataloader):
            img, label = img.to(device), label.to(device)

            # Create "text" label names
            label_names = []
            for l in label:
                label_names.append(CLASS_MAP[l.item()])

            # Create timesteps and noise
            timestep = torch.randint(
                0, cfg["train_args"]["diffusion_steps"] - 1, (img.shape[0],)
            ).long().to(device)
            noise = torch.randn(img.shape).to(device)

            # Add noise to training sample
            noisy_img = noise_scheduler.add_noise(img, noise, timestep)
            # Conditioned noise prediction with input class labels
            pred = model(noisy_img, timestep, label_names)
            loss = loss_fn(pred, noise)
            if hasattr(model, "align_loss"):
                loss += model.align_loss  # Predict noise

            epoch_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_epoch_loss = np.mean(epoch_losses)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epochs": epoch + 1,
                    "loss": avg_epoch_loss
                }
                , cfg["train_args"]["save_ckpt_name"]
            )
        print(f"Loss at epoch {epoch + 1}: {avg_epoch_loss}")
        writer.add_scalar("Training Loss", avg_epoch_loss, epoch + 1)
        if hasattr(model, "align_loss"):
            writer.add_scalar("Alignment loss", model.align_loss.item(), epoch + 1)
        lr_scheduler.step()
    
    print("Training complete.")


def generate_image(
    model,
    noise_scheduler,
    class_label,
    num_imgs,
    cfg,
    device,
    base_imgs,
    fname=""
):
    print("Generating images...")
    model = load_model(model, cfg["train_args"]["save_ckpt_name"])
    
    # Image generation
    noisy_imgs = generate_img(
        model,
        cfg["generator"]["out_ch"],
        noise_scheduler,
        class_label,
        cfg["generator"]["img_sz"],
        num_imgs,
        device
    )
    fname = cfg["generator"]["generations_fname"]
    visualize_img(
        noisy_imgs,
        base_imgs,
        fname,
        CLASS_MAP[class_label],
        cfg["generator"]["generations_folder"]
    )


def main(args, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_mapping = CLASS_MAP

    if args.use_feature_alignment:
        model = FeatureAlignedDiffusion(
            expert_ckpt="r50_expert.pt"
        )
    else:
        model = PreTrainedDiffusion()

    model = model.to(device)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg["train_args"]["diffusion_steps"],
        beta_schedule="squaredcos_cap_v2"
    )

    train_dataloader, _ = get_kather_dataloader(
        cfg["generator"]["img_sz"],
        cfg["generator"]["batch_sz"],
        args.dataset_folder,
        use_grayscale=True if cfg["generator"]["in_ch"] == 1 else False
    )

    if args.do_train:
        train(
            model,
            train_dataloader,
            noise_scheduler,
            cfg,
            device
        )

    # Load checkpoint before generations
    print(f"Loading model checkpoint...")
    model = load_model(model, cfg["train_args"]["save_ckpt_name"])

    for gen_class in range(0, 7):
        print(f"Generating images for class {gen_class}...")
        assert gen_class in class_mapping.keys()
        base_imgs = retrieve_db_images(train_dataloader, gen_class, args.num_img_gens)
        generate_image(
            model,
            noise_scheduler,
            gen_class,
            args.num_img_gens,
            cfg,
            device,
            base_imgs,
        )
        print("Generation complete!")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--do-train", action="store_true", help="Perform training")
    argparser.add_argument("--config", type=str, help="Config yaml with model specifications")
    argparser.add_argument("--num-img-gens", type=int, help="Number of images to generate")
    argparser.add_argument("--dataset-folder", default="Kather_texture_2016_image_tiles_5000",
                           help="Path to Kather dataset")
    argparser.add_argument("--use-feature-alignment", action="store_true", help="Use feature aligned diffusion")
    args = argparser.parse_args()

    cfg = load_yaml(args.config)
    main(args, cfg)
