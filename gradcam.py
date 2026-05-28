import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from config import FRAMES_DIR, HEATMAPS_DIR, IMAGE_SIZE, MODEL_DIR, ensure_dirs
from model import TemporalAvgClassifier


class GradCAM:
    """Grad-CAM implementation for MobileNetV2.

    Hooks into the last convolutional layer to produce spatial attention heatmaps.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """Generate Grad-CAM heatmap for the given input.

        Args:
            input_tensor: (1, 3, 224, 224) input image
            class_idx: target class (0 or 1). If None, uses predicted class.

        Returns:
            heatmap: (224, 224) numpy array normalized to [0, 1]
        """
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = 1  # always show what drives "hand-raise" prediction

        self.model.zero_grad()
        # For binary output, use the single logit directly
        target = output[0] if len(output.shape) == 1 else output[0, 0]
        target.backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # only positive contributions

        # Resize to input size
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def get_mobilenet_for_gradcam(device):
    """Get MobileNetV2 configured for Grad-CAM (with gradients enabled)."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model = model.to(device)
    model.eval()
    return model


def overlay_heatmap(image, heatmap, alpha=0.4):
    """Overlay heatmap on original image."""
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    if image.shape[:2] != heatmap.shape:
        heatmap_colored = cv2.resize(heatmap_colored, (image.shape[1], image.shape[0]))

    overlay = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)
    return overlay


def generate_gradcam_heatmaps(num_samples=10):
    """Generate Grad-CAM heatmaps for sample frames."""
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MobileNetV2 for Grad-CAM
    mobilenet = get_mobilenet_for_gradcam(device)
    target_layer = mobilenet.features[-1]  # last conv block
    grad_cam = GradCAM(mobilenet, target_layer)

    # Load the trained classifier to know which frames are interesting
    classifier = TemporalAvgClassifier().to(device)
    classifier_path = os.path.join(MODEL_DIR, "best_model.pth")
    if os.path.exists(classifier_path):
        classifier.load_state_dict(
            torch.load(classifier_path, map_location=device, weights_only=True)
        )
    classifier.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Find sample frames (mix of positive and negative)
    video_dirs = sorted([d for d in os.listdir(FRAMES_DIR)
                         if os.path.isdir(os.path.join(FRAMES_DIR, d))])

    samples_generated = 0

    for video_id in video_dirs:
        if samples_generated >= num_samples:
            break

        frames_dir = os.path.join(FRAMES_DIR, video_id)
        labels_path = os.path.join(frames_dir, "labels.npy")
        if not os.path.exists(labels_path):
            continue

        labels = np.load(labels_path)
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

        # Pick one positive and one negative frame per video
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]

        sample_indices = []
        if len(pos_indices) > 0:
            sample_indices.append(pos_indices[len(pos_indices) // 2])
        if len(neg_indices) > 0:
            sample_indices.append(neg_indices[len(neg_indices) // 2])

        for frame_idx in sample_indices:
            if samples_generated >= num_samples:
                break
            if frame_idx >= len(frame_files):
                continue

            frame_path = os.path.join(frames_dir, frame_files[frame_idx])
            img = Image.open(frame_path).convert("RGB")
            img_array = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)))

            # Prepare input
            input_tensor = transform(img).unsqueeze(0).to(device)
            input_tensor.requires_grad_(True)

            # Generate Grad-CAM
            heatmap = grad_cam.generate(input_tensor)

            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # Original image
            axes[0].imshow(img_array)
            axes[0].set_title(f"Original (Label: {'Raise' if labels[frame_idx] else 'No Raise'})")
            axes[0].axis("off")

            # Heatmap alone
            axes[1].imshow(heatmap, cmap="jet")
            axes[1].set_title("Grad-CAM Heatmap")
            axes[1].axis("off")

            # Overlay
            overlay = overlay_heatmap(img_array, heatmap)
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay")
            axes[2].axis("off")

            plt.suptitle(f"Video: {video_id}, Frame: {frame_idx}", fontsize=10)
            plt.tight_layout()

            save_path = os.path.join(
                HEATMAPS_DIR,
                f"gradcam_{video_id}_frame{frame_idx:05d}.png"
            )
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

            samples_generated += 1
            print(f"  Saved heatmap: {os.path.basename(save_path)}")

    print(f"Generated {samples_generated} Grad-CAM heatmaps in {HEATMAPS_DIR}")


if __name__ == "__main__":
    generate_gradcam_heatmaps()
