from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import base64
import smtplib, ssl
from email.message import EmailMessage
import tempfile
from flask_cors import CORS  # Import CORS for cross-origin requests
import torch
import cv2
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import datetime


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Email configurations
port = 465  # For SSL
smtp_server = "smtp.gmail.com"
sender_email = "korankye2004@gmail.com"
password = "rofd yqmf wcxy rtza"

# Define class labels
classes = ['pituitary', 'bone fractured', 'healthy brain', 'adenocarcinoma', 'glioma', 'benign', 'breast cancer positive', 'bone not fractured', 'squamous cell carcinoma', 'meningioma', 'breast cancer negative']

# Define image transformations (same as validation set)
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


# Load the trained model
comprehensive_model = torch.load("comprehensive_model.pth", weights_only=False )
comprehensive_model.eval()  # Set to evaluation mode

def base64_to_image(uploaded_image):
    # Remove the data URL prefix if present
    if ',' in uploaded_image:
        uploaded_image = uploaded_image.split(',')[1]
    
    # Decode the base64 string
    image_bytes = base64.b64decode(uploaded_image)
    
    # Create a temporary file with .jpg extension
    image_path = "temp_image.jpg"
    
    # Save the image to the temporary path
    with open(image_path, 'wb') as f:
        f.write(image_bytes)
    
    return image_path

def predict(image_path, diagnosisArea):
    """
    Predict brain tumor classification for multiple image types
    Supports: jpg, jpeg, png, bmp, tiff
    """
    # Supported image extensions
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    try:
        # Validate file extension
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format. Supported formats: {', '.join(SUPPORTED_FORMATS)}")

        # Load image using PIL first (handles multiple formats better)
        pil_image = Image.open(image_path)
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        image = pil_image
        
        # Apply transformations
        transformed = transform(image)
        # Add batch dimension and ensure it's the right shape
        image_tensor = transformed.unsqueeze(0)

        # Move tensor to same device as model
        device = next(comprehensive_model.parameters()).device
        image_tensor = image_tensor.to(device)

        # Perform inference
        with torch.no_grad():
            output = comprehensive_model(image_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence = probabilities[0, predicted_class].item()

        # Return detailed results
        # Filter classes based on diagnosisArea
        if diagnosisArea == "Brain Tumors":
            relevant_classes = ['pituitary', 'healthy brain', 'glioma', 'meningioma']
            filtered_probabilities = {
            cls: probabilities[0, classes.index(cls)].item()
            for cls in relevant_classes
            }
            
            # Find the highest probability among relevant classes
            relevant_probs = torch.tensor([probabilities[0, classes.index(cls)] for cls in relevant_classes])
            filtered_predicted_class = relevant_classes[torch.argmax(relevant_probs).item()]
            filtered_confidence = torch.max(relevant_probs).item()
            
            result = {
            'diagnosis': filtered_predicted_class,
            'confidence': filtered_confidence,
            'probabilities': filtered_probabilities
            }

        elif diagnosisArea == "Breast Cancer":
            relevant_classes = ['breast cancer negative', 'breast cancer positive']
            filtered_probabilities = {
            cls: probabilities[0, classes.index(cls)].item()
            for cls in relevant_classes
            }
            
            # Find the highest probability among relevant classes
            relevant_probs = torch.tensor([probabilities[0, classes.index(cls)] for cls in relevant_classes])
            filtered_predicted_class = relevant_classes[torch.argmax(relevant_probs).item()]
            filtered_confidence = torch.max(relevant_probs).item()
            
            result = {
            'diagnosis': filtered_predicted_class,
            'confidence': filtered_confidence,
            'probabilities': filtered_probabilities
            }

        elif diagnosisArea == "Bone Fractures":
            relevant_classes = ['bone fractured', 'bone not fractured']
            filtered_probabilities = {
            cls: probabilities[0, classes.index(cls)].item()
            for cls in relevant_classes
            }
            
            # Find the highest probability among relevant classes
            relevant_probs = torch.tensor([probabilities[0, classes.index(cls)] for cls in relevant_classes])
            filtered_predicted_class = relevant_classes[torch.argmax(relevant_probs).item()]
            filtered_confidence = torch.max(relevant_probs).item()
            
            result = {
            'diagnosis': filtered_predicted_class,
            'confidence': filtered_confidence,
            'probabilities': filtered_probabilities
            }

        elif diagnosisArea == "Lung Cancer":
            relevant_classes = ['adenocarcinoma', 'benign', 'squamous cell carcinoma']
            filtered_probabilities = {
            cls: probabilities[0, classes.index(cls)].item()
            for cls in relevant_classes
            }
            
            # Find the highest probability among relevant classes
            relevant_probs = torch.tensor([probabilities[0, classes.index(cls)] for cls in relevant_classes])
            filtered_predicted_class = relevant_classes[torch.argmax(relevant_probs).item()]
            filtered_confidence = torch.max(relevant_probs).item()
            
            result = {
            'diagnosis': filtered_predicted_class,
            'confidence': filtered_confidence,
            'probabilities': filtered_probabilities
            }        
        else:
            # Default behavior for other areas or when no area is specified
            result = {
            'diagnosis': classes[predicted_class],
            'confidence': confidence,
            'probabilities': {
                classes[i]: probabilities[0, i].item() 
                for i in range(len(classes))
            }
            }
        
        # Print formatted results
        print(f"\nDiagnosis Results:")

        print(f"Classification: {result['diagnosis']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nClass Probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"{cls}: {prob:.2%}")
            
        return result

    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found")
        return None
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def generate_heatmap(image_path, model=comprehensive_model, target_layer=comprehensive_model.layer4[-1]):
    """Generate a heatmap using GradCAM and return it as a base64 string"""
    try:
        # Set up GradCAM
        class GradCAM:
            def __init__(self, model, target_layer):
                self.model = model
                self.target_layer = target_layer
                self.gradients = None
                self.features = None
                
                # Register hooks
                self.target_layer.register_forward_hook(self.save_features)
                self.target_layer.register_full_backward_hook(self.save_gradients)
            
            def save_features(self, module, input, output):
                self.features = output
                
            def save_gradients(self, module, grad_input, grad_output):
                self.gradients = grad_output[0]
                
            def generate(self, input_image):
                # Forward pass
                model_output = self.model(input_image)
                predicted_class = torch.argmax(model_output)
                
                # Backward pass
                model_output[:, predicted_class].backward()
                
                # Generate CAM
                pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
                for i in range(self.features.shape[1]):
                    self.features[:, i, :, :] *= pooled_gradients[i]
                
                cam = torch.mean(self.features, dim=1).squeeze()
                cam = torch.maximum(cam, torch.zeros_like(cam))  # ReLU
                cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize
                
                return cam.detach(), predicted_class

        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]

        # Convert NumPy array to PIL Image
        pil_image = Image.fromarray(image)

        # Apply transformations
        transformed = transform(pil_image)
        input_tensor = transformed.unsqueeze(0)

        # Initialize GradCAM
        grad_cam = GradCAM(model, target_layer)

        # Generate heatmap
        cam, predicted_class = grad_cam.generate(input_tensor)
        cam = cam.cpu().numpy()

        # Resize heatmap to match original image size
        cam = cv2.resize(cam, (original_size[1], original_size[0]))

        # Normalize and apply color map
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Blend original image with heatmap
        alpha = 0.5
        superimposed = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        
        # Convert superimposed image to base64
        superimposed_pil = Image.fromarray(superimposed)
        buffer = io.BytesIO()
        superimposed_pil.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{img_str}"

    except Exception as e:
        print(f"Error applying GradCAM: {str(e)}")
        return None

@app.route('/process-image', methods=['POST'])
def handle_file_upload():
    try:
        data = request.get_json()
        
        # Check if required data is provided
        if 'image' not in data:
            return jsonify({"status": "failure", "message": "No image data provided"}), 400
            
        if 'diagnosisArea' not in data:
            return jsonify({"status": "failure", "message": "No diagnosis area specified"}), 400
        
        # Extract data
        image_data = data.get('image')
        diagnosis_area = data.get('diagnosisArea')
        
        # Save the image to a temporary file
        temp_image_path = base64_to_image(image_data)
        
        # Get prediction
        prediction_result = predict(temp_image_path, diagnosis_area)
        
        # Get the heatmap image as base64
        heatmap_base64 = generate_heatmap(temp_image_path)
        
        # Clean up temporary file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        if prediction_result:
            return jsonify({
                "status": "success", 
                "message": "Image processed successfully", 
                "result": prediction_result,
                "heatmap": heatmap_base64
            }), 200
        else:
            return jsonify({"status": "failure", "message": "Failed to process image"}), 500
            
    except Exception as e:
        return jsonify({"status": "failure", "message": f"Error: {str(e)}"}), 500

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    try:
        data = request.get_json()
        
        if 'pdf_data' not in data:
            return jsonify({"status": "failure", "message": "No PDF data provided"}), 400
            
        pdf_data = data.get('pdf_data')
        filename = data.get('filename', 'diagnostics-report.pdf')
        
        # Store the PDF data in a temporary file to verify it
        temp_file = None
        try:
            # Remove base64 prefix if present
            if "base64," in pdf_data:
                pdf_data = pdf_data.split("base64,")[1]
                
            # Create a temporary file to save the decoded data
            file_extension = ".pdf"
            temp_file = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False)
            
            with open(temp_file.name, "wb") as f:
                f.write(base64.b64decode(pdf_data))
                
            # Store the path in a session variable or database
            # For simplicity, we'll just return the temp file path
            # In production, you would store this more securely
            pdf_id = os.path.basename(temp_file.name)
            
            # Don't delete the temp file yet
            return jsonify({
                "status": "success", 
                "message": "PDF uploaded successfully", 
                "pdf_id": pdf_id
            }), 200
                
        except Exception as e:
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise e
            
    except Exception as e:
        return jsonify({"status": "failure", "message": f"Error: {str(e)}"}), 500

@app.route('/submit-scan', methods=['POST'])
def handle_scan_submission():
    try:
        data = request.get_json()
        
        # Extract data
        if 'receiver_emails' not in data:
            return jsonify({"status": "failure", "message": "No receiver emails provided"}), 400
            
        receiver_emails = data.get('receiver_emails', [])
        scan_type = data.get('scan_type', 'Scan')
        diagnosis_area = data.get('diagnosis_area', 'General')
        subject = data.get('subject', '')
        description = data.get('description', '')
        pdf_data = data.get('pdf_data', '')  # Get PDF data if provided
        pdf_id = data.get('pdf_id', '')      # Get PDF ID if provided
        
        # If we have a PDF ID, try to find the corresponding file
        pdf_path = None
        if pdf_id:
            possible_path = os.path.join(tempfile.gettempdir(), pdf_id)
            if os.path.exists(possible_path):
                pdf_path = possible_path
                
        # If we have direct PDF data or a PDF path, process it
        if pdf_data or pdf_path:
            if pdf_data:
                success = process_scan_data(receiver_emails, scan_type, diagnosis_area, pdf_data, subject, description)
            else:
                success = send_email_with_pdf(receiver_emails, scan_type, diagnosis_area, subject, description, pdf_path)
        else:
            # Fallback to just text email
            success = send_email_without_attachment(receiver_emails, scan_type, diagnosis_area, subject, description)
        
        if success:
            # Clean up the temporary file if it exists
            if pdf_path and os.path.exists(pdf_path):
                try:
                    os.unlink(pdf_path)
                except:
                    pass  # Ignore errors when cleaning up
                    
            return jsonify({"status": "success", "message": "Results shared successfully"}), 200
        else:
            return jsonify({"status": "failure", "message": "Failed to send email"}), 500
            
    except Exception as e:
        return jsonify({"status": "failure", "message": f"Error: {str(e)}"}), 500

def send_email_with_pdf(receiver_emails, scan_type, diagnosis_area, subject, description, pdf_path):
    """Send email with a PDF attachment from a file path"""
    try:
        # Message
        message = EmailMessage()
        message["From"] = sender_email
        message["To"] = ", ".join(receiver_emails) if isinstance(receiver_emails, list) else receiver_emails
        
        # Use custom subject if provided, otherwise use default
        email_subject = subject if subject else f"{diagnosis_area} {scan_type} Diagnostics AI Diagnosis Results"
        message["Subject"] = email_subject
        
        # Use custom description if provided, otherwise use default
        email_body = description if description else "Please find attached the scan results from Diagnostics AI."
        message.set_content(email_body)

        # Attach the PDF file
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as file:
                file_data = file.read()
                
                # Generate a more descriptive filename
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"diagnostics_ai_{diagnosis_area.lower().replace(' ', '_')}_{timestamp}.pdf"
                
                # Add the attachment with proper MIME headers
                message.add_attachment(
                    file_data,
                    maintype="application",
                    subtype="pdf",
                    filename=filename,
                    disposition="attachment"
                )
        else:
            raise FileNotFoundError("The specified PDF file does not exist.")

        # Send Email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.send_message(message)
        return True
    except Exception as e:
        print(f"Error sending email with PDF: {str(e)}")
        return False

def send_email(receiver_emails, scan_type, diagnosis_area, uploaded_file, subject=None, description=None):
    """Email sending functionality with improved PDF attachment handling"""
    try:
        # Message
        message = EmailMessage()
        message["From"] = sender_email
        message["To"] = ", ".join(receiver_emails) if isinstance(receiver_emails, list) else receiver_emails
        
        # Use custom subject if provided, otherwise use default
        email_subject = subject if subject else f"{diagnosis_area} {scan_type} Diagnostics AI Diagnosis Results"
        message["Subject"] = email_subject
        
        # Use custom description if provided, otherwise use default
        email_body = description if description else "Please find attached the scan results from Diagnostics AI."
        message.set_content(email_body)

        # Attach the file
        if os.path.exists(uploaded_file):
            with open(uploaded_file, "rb") as file:
                file_data = file.read()
                file_name = os.path.basename(uploaded_file)
                file_ext = os.path.splitext(file_name)[1][1:]
                
                # Generate a more descriptive filename with timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                descriptive_filename = f"diagnostics_ai_{diagnosis_area.lower().replace(' ', '_')}_{timestamp}.{file_ext}"
                
                # Determine maintype based on extension
                maintype = "image" if file_ext.lower() in ["jpg", "jpeg", "png", "gif"] else "application"
                subtype = file_ext if file_ext else "pdf" if file_ext.lower() == "pdf" else "octet-stream"
                
                message.add_attachment(
                    file_data,
                    maintype=maintype,
                    subtype=subtype,
                    filename=descriptive_filename,
                    disposition="attachment"
                )
        else:
            raise FileNotFoundError("The specified file does not exist.")

        # Send Email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.send_message(message)
        
        return True
    except Exception as e:
        print(f"Error sending email with attachment: {str(e)}")
        return False

def process_scan_data(receiver_emails, scan_type, diagnosis_area, pdf_data, subject=None, description=None):
    """
    Process the scan data and send email with PDF attachment
    """
    try:
        print(f"Processing scan with the following details:")
        print(f"Receiver emails: {receiver_emails}")
        print(f"Scan type: {scan_type}")
        print(f"Diagnosis area: {diagnosis_area}")
        print(f"PDF data length: {len(pdf_data) if pdf_data else 0} characters")
        
        # Decode base64 data and save to a temporary file
        temp_file = None
        if pdf_data:
            # Remove base64 prefix if present (e.g., "data:application/pdf;base64,")
            if "base64," in pdf_data:
                pdf_data = pdf_data.split("base64,")[1]
            
            # Create a temporary file to save the decoded data
            file_extension = ".pdf"
            temp_file = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False)
            
            with open(temp_file.name, "wb") as f:
                f.write(base64.b64decode(pdf_data))
                
            # Send email with the file using improved function
            success = send_email(receiver_emails, scan_type, diagnosis_area, temp_file.name, subject, description)
            
            # Clean up the temporary file
            os.unlink(temp_file.name)
            
            return success
        else:
            return False
    except Exception as e:
        print(f"Error processing scan: {str(e)}")
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        return False

def apply_gradcam(image_path, model=comprehensive_model, target_layer=comprehensive_model.layer4[-1]):
    # Set up GradCAM
    class GradCAM:
        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer
            self.gradients = None
            self.features = None
            
            # Register hooks
            self.target_layer.register_forward_hook(self.save_features)
            self.target_layer.register_full_backward_hook(self.save_gradients)
        
        def save_features(self, module, input, output):
            self.features = output
            
        def save_gradients(self, module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        def generate(self, input_image):
            # Forward pass
            model_output = self.model(input_image)
            predicted_class = torch.argmax(model_output)
            
            # Backward pass
            model_output[:, predicted_class].backward()
            
            # Generate CAM
            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
            for i in range(self.features.shape[1]):
                self.features[:, i, :, :] *= pooled_gradients[i]
            
            cam = torch.mean(self.features, dim=1).squeeze()
            cam = torch.maximum(cam, torch.zeros_like(cam))  # ReLU
            cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize
            
            return cam.detach(), predicted_class  # Detach here

    try:
        # Load image using OpenCV
        image = cv2.imread(image_path)  # OpenCV loads as BGR NumPy array
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        original_size = image.shape[:2]  # Store original size (H, W)

        # Convert NumPy array to PIL Image (required for torchvision transforms)
        pil_image = Image.fromarray(image)

        # Apply transformations
        transformed = transform(pil_image)  # Now it's a PyTorch tensor (C, H, W)
        input_tensor = transformed.unsqueeze(0)  # Add batch dimension (1, C, H, W)

        # Initialize GradCAM
        grad_cam = GradCAM(model, target_layer)

        # Generate heatmap
        cam, predicted_class = grad_cam.generate(input_tensor)
        cam = cam.cpu().numpy()  # Convert to NumPy array

        # Resize heatmap to match original image size
        cam = cv2.resize(cam, (original_size[1], original_size[0]))  # (W, H) format

        # Normalize and apply color map
        heatmap = np.uint8(255 * cam)  # Normalize to [0, 255]
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Blend original image with heatmap
        alpha = 0.5
        superimposed = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        # Display results
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')

        ax2.imshow(heatmap)
        ax2.set_title('GradCAM Heatmap')
        ax2.axis('off')

        ax3.imshow(superimposed)
        ax3.set_title('Superimposed Image')
        ax3.axis('off')

        plt.suptitle(f'Predicted Class: {classes[predicted_class]}')
        plt.tight_layout()
        plt.show()

        return cam  # Return GradCAM output

    except Exception as e:
        print(f"Error applying GradCAM: {str(e)}")
        return None
    
def image_pipeline(uploaded_image, diagnosisArea, model=comprehensive_model, target_layer=comprehensive_model.layer4[-1], image_path = None):
    # Get image path from base64_to_image
    image_path = base64_to_image(uploaded_image)
    
    # Apply GradCAM using the image path
    apply_gradcam(image_path, model, target_layer)
    
    # Get prediction using the image path
    result = predict(image_path, diagnosisArea)
    
    # Clean up temporary file
    if os.path.exists(image_path):
        os.remove(image_path)
        
    return result

def send_email_without_attachment(receiver_emails, scan_type, diagnosis_area, subject, description):
    """Send email without an attachment, just with the text content"""
    try:
        # Message
        message = EmailMessage()
        message["From"] = sender_email
        message["To"] = ", ".join(receiver_emails) if isinstance(receiver_emails, list) else receiver_emails
        
        # Use custom subject if provided, otherwise use default
        email_subject = subject if subject else f"{diagnosis_area} {scan_type} Diagnostics AI Diagnosis Results"
        message["Subject"] = email_subject
        
        # Use custom description if provided, otherwise use default
        email_body = description if description else "Results shared from our diagnostic platform."
        message.set_content(email_body)

        # Send Email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.send_message(message)
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False



if __name__ == '__main__':
    app.run(debug=True)