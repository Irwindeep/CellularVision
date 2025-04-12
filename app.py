import streamlit as st
import numpy as np

import torch
from torchvision import transforms
from PIL import Image
from cellularvision.postprocessing.utils import get_segmentation_contours
from cellularvision.postprocessing import mean_shift_segmentation

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
resize = transforms.Resize((224, 224))
inv_normalize = transforms.Normalize(
    (-1, -1, -1),
    (2, 2, 2)
)

device = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

st.title("CellularVision: Analysis and Segmentation of Histopathological Tissues")

uploaded_file = st.sidebar.file_uploader(
    "Choose an image file", type=["jpg", "jpeg", "png"]
)
if uploaded_file is not None:
    image = resize(Image.open(uploaded_file))
    
    st.image(image, caption="Uploaded Image")
    image_torch = transform(image).unsqueeze(0)
    st.session_state["image"] = image
    st.session_state["image_torch"] = image_torch
else:
    st.write("Please upload an image file to continue.")

if "image" in st.session_state:
    if st.sidebar.button("Segment with CNN"):
        from cellularvision.models import SegmentationCNN

        model = SegmentationCNN().to(device)
        model.load_state_dict(
            torch.load("model_weights/cnn_weights.pth",
                    map_location=device)
        )
        model.eval()
        with torch.no_grad():
            logits = model(image_torch).squeeze()
        pred = torch.argmax(logits, dim=0).numpy()
        segmentation = get_segmentation_contours(
            np.array(image), pred
        )
        st.image(segmentation, caption="Predicted Segmentation")
        mean_shift = mean_shift_segmentation(logits.numpy())
        segmentation_ms = get_segmentation_contours(
            np.array(image), mean_shift
        )
        st.image(segmentation_ms, caption="Refined Segmentation with Mean Shift")

    if st.sidebar.button("Segment with UNet"):
        from cellularvision.models import UNet

        model = UNet(
            5, [3, 32, 64, 128, 256], 3
        ).to(device)
        model.load_state_dict(
            torch.load("model_weights/unet_weights.pth",
                    map_location=device)
        )
        with torch.no_grad():
            logits = model(image_torch).squeeze()
        pred = torch.argmax(logits, dim=0).numpy()
        segmentation = get_segmentation_contours(
            np.array(image), pred
        )
        st.image(segmentation, caption="Predicted Segmentation")
        mean_shift = mean_shift_segmentation(logits.numpy())
        segmentation_ms = get_segmentation_contours(
            np.array(image), mean_shift
        )
        st.image(segmentation_ms, caption="Refined Segmentation with Mean Shift")

    if st.sidebar.button("Segment with SegNet"):
        from cellularvision.models import SegNet

        model = SegNet(
            5, [2, 2, 1, 1, 1], pretrained=False
        ).to(device)
        model.load_state_dict(
            torch.load("model_weights/segnet_weights.pth",
                    map_location=device)
        )
        model.eval()
        with torch.no_grad():
            logits = model(image_torch).squeeze()
        pred = torch.argmax(logits, dim=0).numpy()
        segmentation = get_segmentation_contours(
            np.array(image), pred
        )
        st.image(segmentation, caption="Predicted Segmentation")
        mean_shift = mean_shift_segmentation(logits.numpy())
        segmentation_ms = get_segmentation_contours(
            np.array(image), mean_shift
        )
        st.image(segmentation_ms, caption="Refined Segmentation with Mean Shift")

    if st.sidebar.button("Segment with PSPNet"):
        from segmentation_models_pytorch import PSPNet

        model = PSPNet(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes= 6
        ).to(device)
        model.load_state_dict(
            torch.load("model_weights/pspnet_weights.pth",
                    map_location=device)
        )
        model.eval()
        with torch.no_grad():
            logits = model(image_torch).squeeze()
        pred = torch.argmax(logits, dim=0).numpy()
        segmentation = get_segmentation_contours(
            np.array(image), pred
        )
        st.image(segmentation, caption="Predicted Segmentation")
        mean_shift = mean_shift_segmentation(logits.numpy())
        segmentation_ms = get_segmentation_contours(
            np.array(image), mean_shift
        )
        st.image(segmentation_ms, caption="Refined Segmentation with Mean Shift")

    if st.sidebar.button("Segment with MaskRCNN"):
        from training_scripts.train_mask_rcnn import load_model

        model = load_model(pretrained=False)
        model.load_state_dict(
            torch.load("model_weights/mask_rcnn_weights.pth",
                    map_location=device)
        )
        model.eval()
        with torch.no_grad():
            logits = model(image_torch)[0]

        pred_mask = np.zeros((224, 224))
        for mask, label, score in zip(
            logits["masks"], logits["labels"], logits["scores"]
        ):
                if score < 0.2:
                    continue
                indices = (mask > 0.0).float().numpy()
                pred_mask[indices.astype(int)] = int(np.argmax(label))
        segmentation = get_segmentation_contours(
            np.array(image), pred_mask
        )
        st.image(segmentation, caption="Predicted Segmentation")
