# streamlit builds the web interface
# streamlit_drawable_canvas is the custom drawing component where users can highlight the image
# PIL.image is for loading and handling images
# torch & torchvision are deep learning libraries and have pretrained models (ResNet)
# cv2 is for resizing, heatmaps, and overlaying images
# os and random is for traversing system files and selecting a random image

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import torch
from torchvision import models, transforms
import numpy as np
import cv2
import os
import random

# shows the app's title at the top of the page
st.title("Do You See What AI Thinks Is Important?")

# includes important information/context for the user
st.markdown("""
This interactive app lets you explore **how humans and AI focus on different parts of an image** for classification.  

**How it works:**  
1. Upload an image or select a random one below.  
2. Use the pen tool to highlight regions in the image that you think are important for identifying the object in the image.  
3. Guess the model’s predicted class for that image.  
4. Click “Run Model and GradCAM” to see how a CNN (ResNet-18 pretrained on ImageNet) interprets the image.  
5. Compare your highlighted regions with the model’s GradCAM heatmap and see your **IoU (Intersection over Union) score**. 
6. Press "Reset Image" to upload another image or select another random image. Make sure that after pressing "Reset Image" you clear any images that were uploaded.

This tool helps you understand **where AI “looks” versus where humans focus**, providing insight into model reasoning, attention alignment, and potential biases.
""")

with st.expander("Why This Matters"):
    st.markdown("""
AI systems in medicine, driving, security, and other critical domains must be **interpretable** and **explainable**.  
By comparing human vs. model attention (what is focused on), we can identify:

- Biased model attention  
- Missing attention on key features  
- Overreliance on background context  
- Whether model reasoning matches human intuition  

Understanding *how* models make decisions is essential for transparency and trust.
""")

with st.expander("What is GradCAM?"):
    st.markdown("""
GradCAM (Gradient-weighted Class Activation Mapping) produces a heatmap that highlights the parts of an image that most influence a convolutional neural network's (CNN) prediction.

**Generally warmer colors indicate regions that are more important for the model:**
- **Red** → Strong model focus  
- **Green / Yellow** → Moderate focus  
- **Blue** → Low / no focus  

""")

with st.expander("What Do You Do?"):
    st.markdown("""
First either upload an image or select a random image.
Then use the pen tool to mark regions that you believe are essential for recognizing the object. 
You can also select the class that you think the image should be classified as (e.g. a picture of a shark could be classified as 'great white shark').
""")
    
with st.expander("GradCAM Threshold Slider"):
    st.markdown("""
This allows you to control which GradCAM pixels the model counts as “important.”

- **Lower threshold** → More regions are included  
- **Higher threshold** → Only the most imporant regions are kept  
Adjusting this changes the IoU score and the GradCAM heatmap produced by the model.
""")

with st.expander("Intersection over Union Score"):
    st.markdown("""
The app computes **Intersection over Union (IoU)** between your drawing and the GradCAM heatmap (after thresholding).

It is calculated by dividing the shared pixels that you and the GradCAM heatmap highlight with the total pixels highlighted by you and the GradCAM heatmap.
It ranges from 0 to 1 (0 meaning no overlap and 1 being perfect overlap) and quantifies how similarly you and the model “look” at the image.
""")

with st.expander("Overlap Map"):
    st.markdown("""
After comparing your drawing with GradCAM, the app shows a blended visualization of 3 regions:

- **🟩 Green Region – Agreement** (both you and the model highlighted it)
- **🟥 Red Region – Model only** (the model highlighted it but you didn’t)
- **🟦 Blue Region – User only** (you highlighted it but the model didn't)

These colors help you see where reasoning aligns or diverges.
""")
    
with st.expander("Model & Dataset Details"):
    st.markdown("""
This app uses a **ResNet-18** model from the `torchvision.models` library.  
It is **pretrained on the ImageNet dataset**, which contains 1,000 object categories.

**Important notes:**
- Random images in this app come from a small ImageNet dataset (https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000).  
- When guessing the prediction, make sure you are selecting a class from the **ImageNet class list** (https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt).  
- The GradCAM heatmap reflects what ResNet-18 considers important for its ImageNet classification.
""")
    
st.markdown('''Have Fun!''')


@st.cache_data
def load_imagenet_classes():
    '''
    Downloads ImageNet classes (1000 class names) from Pytorch Github
    Split the txt file from Github into lines and return a list of class strings
    Cache data prevents re-downloading for every rerun
    '''
    import requests
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    classes = [s.strip() for s in response.text.splitlines()]
    return classes

imagenet_classes = load_imagenet_classes()

@st.cache_resource
def load_model():
    '''
    Load pre-trained ResNet18 on ImageNet
    Eval puts the model in inference mode
    Cache again similar to load_imagenet_classes above
    '''
    model = models.resnet18(pretrained=True)
    model.eval()
    return model

model = load_model()

def generate_gradcam(model, input_tensor, class_idx):
    '''
    This function computes GradCAM for a given model (ResNet18 rn) and a target class
    It uses forward and backward hooks to capture activations from a convolutional layer
    and the gradients that flow back from the selected class
    '''

    # these will contain the gradients and forward activations of the target layer
    gradients = None
    activations = None

    def save_gradient(module, grad_input, grad_output):
        '''
        This is a hook called during backward() to get the gradients
        '''
        nonlocal gradients
        gradients = grad_output[0]

    def forward_hook(module, input, output):
        '''
        This is a hook called during forward() to get the feature maps 
        '''
        nonlocal activations
        activations = output

    # select the final convolutional layer in the model (ResNet18) 
    # attach the hooks
    target_layer = model.layer4[1].conv2
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(save_gradient)

    # forward pass 
    output = model(input_tensor)

    # we want the gradients with respect to the particular class output
    model.zero_grad()

    # make a one-hot encoded vector that selects only the target class 
    one_hot = torch.zeros_like(output)
    one_hot[0, class_idx] = 1

    # backpropagate to get the gradients for the target class 
    output.backward(gradient=one_hot)

    # calculate channel-wise gradient importance weights
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # multiply each activation channel by its respective gradient weight
    for i in range(activations.shape[1]):
        activations[0, i, :, :] *= pooled_gradients[i]

    # average over the activation channels to get a heatmap 
    heatmap = torch.mean(activations, dim=1).detach().numpy()[0]

    # effectively apply ReLU (0 and positive)
    heatmap = np.maximum(heatmap, 0)

    # normalize to [0,1]
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap

# load the random example images
# gathers paths to all image files
# ImageNet_Random folder has images from a kaggle ImageNet dataset that I downloaded
example_paths = []
for root, dirs, files in os.walk("ImageNet_Random"):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            example_paths.append(os.path.join(root, f))

# add a reset button at the top of the page
# after you are done predicting and seeing gradcam for 1 image, this button resets the page 
# cleans the selected image, mask, and guess 
# easy to upload or choose a new random image after it
if st.button("Reset (New Image)"):
    st.session_state["selected_image"] = None
    st.session_state["user_mask"] = None
    st.session_state["user_guess"] = None
    st.experimental_rerun()

# initialize the selected image (image either uploaded or randomly selected)
if "selected_image" not in st.session_state:
    st.session_state["selected_image"] = None

# give the user the option of uploading an image 
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    st.session_state["selected_image"] = Image.open(uploaded_file).convert("RGB")

# give the user the option of using an image from the ImageNet_Random folder 
if st.button("Use a Random Image"):
    # error handling code incase file paths don't have images
    if len(example_paths) == 0:
        st.error("No images found in 'ImageNet_Random/' folder or subfolders!")
    else:
        random_path = random.choice(example_paths)
        st.session_state["selected_image"] = Image.open(random_path).convert("RGB")

# make sure user selects an image before proceeding
img = st.session_state["selected_image"]
if img is None:
    st.warning("Upload an image or select a random image to start.")
    st.stop()

# the first row shows the original image
st.image(img, caption="Selected Image", use_column_width=True)

# the second row contains the same image but the user can draw on it here (canvas to draw)
st.write("Color the regions that you think are important for classification:")

# this slider allows the user to adjust the stroke width (size) of the pen tool 
# a larger stroke width increases the size of the pen (smaller makes the pen size smaller)
stroke_width = st.slider("Stroke Width", min_value=1, max_value=50, value=30, step=1, help="Controls the brush thickness on the canvas.")

# this slider controls the threshold for which pixels from the model's gradcam count towards the miou score
# a higher value reduces the number of pixels used for miou (more strict, pixels that contributed most to prediction)
# a smaller value increases the number of pixels used for miou (less strict, areas that have a blue overlay may contribute to the miou)
threshold = st.slider("GradCAM Threshold Value", min_value=0.0, max_value=1.0, value=0.5, step=0.05, help="Higher value is stricter (only highest-intensity GradCAM regions are counted)")

# Canvas size based on original image aspect ratio but with max width of 800
max_width = 800
canvas_width = min(max_width, img.width)
canvas_height = int(canvas_width * img.height / img.width)

# overlay the drawing on the image (highlight is in red)
canvas_result = st_canvas(fill_color="rgba(255,0,0,0.3)", stroke_width=stroke_width, stroke_color="red", background_image=img, update_streamlit=True, height=canvas_height, width=canvas_width, drawing_mode="freedraw", key="canvas",)

# once mask is drawn, convert it to a binary mask
if canvas_result.image_data is not None:

    # use alpha channel to detect where the user drew (alpha > 0 equates to a highlighted pixel)
    mask = np.array(canvas_result.image_data[:, :, 3] > 0, dtype=np.uint8)

    if mask.sum() > 0:
        # resize the mask back to the original image size 
        mask_resized = cv2.resize(mask.astype(np.uint8), (img.width, img.height), interpolation=cv2.INTER_NEAREST)
        st.session_state["user_mask"] = mask_resized
    else:
        # if no drawing is detected 
        st.session_state["user_mask"] = None

# users can also guess what class the model will output for the image
st.write("Guess the model's predicted class before running the model:")

# filtered classes includes a short list of the classes with the word entered in search_text
# way too many classes to just have a single dropdown or expect the user to perfectly enter the class name 
# developed this system to make predicting the model's prediction easier
search_text = st.text_input("Type to search for a class", value="")
filtered_classes = [c for c in imagenet_classes if search_text.lower() in c.lower()]

# display a dropdown if classes are found based on search_text input
if filtered_classes:
    user_guess = st.selectbox("Select class from filtered options:", filtered_classes)
    st.session_state["user_guess"] = user_guess
else:
    st.write("No matching classes found.")
    st.session_state["user_guess"] = None

# run the model to see the prediction and GradCAM
run_analysis = st.button("Run Model and GradCAM")

if run_analysis:
    # if the user presses the run moel and GradCAM button but they did not draw anything 
    # show a warning and wait for them to draw and press button again
    if "user_mask" not in st.session_state or st.session_state["user_mask"] is None:
        st.warning("Please draw on the image before running analysis!")
    else:
        user_mask_raw = st.session_state["user_mask"]

        # standard ImageNet preprocessing 
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        img_tensor = preprocess(img).unsqueeze(0)

        # perform forward pass 
        output = model(img_tensor)
        pred_class = output.argmax(dim=1).item()
        pred_name = imagenet_classes[pred_class]

        # gets the GradCAM heatmap 
        heatmap = generate_gradcam(model, img_tensor, pred_class)
        
        # resize it to match the image
        heatmap_resized = cv2.resize(heatmap, (img.width, img.height))

        # colorize the heatmap with OpenCV colormap 
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # blend GradCAM heatmap onto original image 
        img_np = np.array(img)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

        # binary mask of GradCAM (only pixels that are greater than the threshold value set earlier by user with slider)
        gradcam_mask_bin = (heatmap_resized > threshold).astype(np.uint8)

        # intersection over union calculation (pixels that are in both user mask and gradcam mask/pixels from both masks together)
        intersection = np.logical_and(user_mask_raw, gradcam_mask_bin).sum()
        union = np.logical_or(user_mask_raw, gradcam_mask_bin).sum()
        iou = intersection / (union + 1e-8)


        # display the results in two columns (GradCAM heatmap and user mask comparison)
        col1, col2 = st.columns(2)

        # GradCAM overlay
        with col1:
            st.image(overlay, caption=f"GradCAM Overlay", use_column_width=True)

        # user mask vs. GradCAM comparison
        with col2:
            combined_mask = np.zeros_like(img_np)

            # green represents correct overlap
            intersection_mask = np.logical_and(user_mask_raw, gradcam_mask_bin)
            combined_mask[intersection_mask] = [0, 255, 0]

            # red represents areas that the user marked but GradCAM did not 
            user_only_mask = np.logical_and(user_mask_raw, np.logical_not(gradcam_mask_bin))
            combined_mask[user_only_mask] = [255, 0, 0]

            # blue represents areas that GradCAM marked but the user did not 
            gradcam_only_mask = np.logical_and(np.logical_not(user_mask_raw), gradcam_mask_bin)
            combined_mask[gradcam_only_mask] = [0, 0, 255]

            blended_display = cv2.addWeighted(img_np, 0.6, combined_mask, 0.4, 0)
            st.image(blended_display, caption="Blended Intersection & Mismatches\nGreen = Correct, Red = Extra, Blue = Missed", use_column_width=True)

        # display the predicted class and user IoU 
        st.write(f"Predicted class: **{pred_class} ({pred_name})**")
        st.write(f"You vs GradCAM IoU: **{iou:.3f}**")

        # write the user's guess and say whether it is right or wrong compared to the model prediction
        if "user_guess" in st.session_state and st.session_state["user_guess"]:
            guess_match = st.session_state["user_guess"] == pred_name
            st.write(f"Your guess: **{st.session_state['user_guess']}** → {'Correct :)' if guess_match else 'Incorrect :('}")
