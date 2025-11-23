# XAI_Final_Project_GradCAM_App
Streamlit app for XAI Final Project. Through this app, a user can highlight regions of an image that they think is important for model classification and compare it against the GradCAM heatmap from a CNN (ResNet18 trained on ImageNet from torchvision models) using Intersection over Union (IoU). 

The user has the option of uploading an image of their own or using a random image that comes from the ImageNet_Random folder (an ImageNet dataset from https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000). Once they select an image, they can draw the regions that they think are important for the model's classification. They also have the option to control the stroke width of the drawing pen and the threshold value. The threshold value controls how many of the pixels of the GradCAM overlay contribute to the IoU calculation. A larger threshold value is more strict so it will include mainly the most important parts (red areas) while a lower threshold will expand the overlay to include less important regions (include blue areas). They can also predict the class that the model will output (comes from ImageNet class list: https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt). 

Once they are happy with their prediction and highlighted region(s), then they can run the model to get the model's GradCAM heatmap and prediction. The app calculates the IoU of the GradCAM and their drawings and assigns a score. It also shows the model's predicted class in comparison to the user's. To reset the app to try a new image, users can press the "Reset Image" button. Have fun!

The project can be accessed here: 
https://xai-gradcam-app.streamlit.app


How to Run the Project Locally: 
1. Clone this repository
2. Install the dependecies in requirements.txt
3. Run the streamlit app (streamlit run gradcam_app.py)
4. The app should then open in your local browser

