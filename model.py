# model.py
import torch
import torch.nn as nn
import timm
from sentence_transformers import SentenceTransformer
import torchvision.models as models
from torchvision.models.video import r3d_18


class MultiModalEncoder(nn.Module):
    """Handles encoding for different medical data types"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device

        # Image encoder (CT scans, X-rays, pictures)
        self.image_encoder = timm.create_model(
            'efficientnet_b3', pretrained=True, num_classes=0)

        # Video encoder (CT scans over time, ultrasound videos)
        self.video_encoder = r3d_18(pretrained=True)
        self.video_encoder.fc = nn.Identity()  # Remove final classification layer

        # Text encoder (medical reports, doctor notes)
        self.text_encoder = SentenceTransformer('all-mpnet-base-v2')
        self.text_feature_dim = 768

        # Projection layers to common space
        self.image_proj = nn.Linear(1536, 512)
        self.video_proj = nn.Linear(512, 512)
        self.text_proj = nn.Linear(self.text_feature_dim, 512)

        # Fusion attention
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=512, num_heads=8, batch_first=True)

    def encode_image(self, image):
        """Encode static medical images"""
        features = self.image_encoder(image)
        return self.image_proj(features)

    def encode_video(self, video):
        """Encode medical video sequences"""
        # video shape: (batch, frames, channels, height, width)
        batch_size, num_frames = video.shape[:2]
        video = video.view(batch_size * num_frames, *video.shape[2:])
        frame_features = self.image_encoder(video)
        frame_features = frame_features.view(batch_size, num_frames, -1)
        features = self.video_encoder(frame_features)
        return self.video_proj(features)

    def encode_text(self, text):
        """Encode medical text reports"""
        with torch.no_grad():
            embeddings = self.text_encoder.encode(text, convert_to_tensor=True)
        return self.text_proj(embeddings)

    def forward(self, image=None, video=None, text=None):
        """Encode multi-modal medical data"""
        embeddings = []

        if image is not None:
            img_emb = self.encode_image(image)
            embeddings.append(img_emb.unsqueeze(1))

        if video is not None:
            vid_emb = self.encode_video(video)
            embeddings.append(vid_emb.unsqueeze(1))

        if text is not None:
            txt_emb = self.encode_text(text)
            embeddings.append(txt_emb.unsqueeze(1))

        # Concatenate all embeddings
        combined = torch.cat(embeddings, dim=1)

        # Attention-based fusion
        query = torch.mean(combined, dim=1, keepdim=True)
        fused, _ = self.fusion_attention(query, combined, combined)
        return fused.squeeze(1)


class DiagnosticModel(nn.Module):
    """AI diagnostic system for medical analysis"""

    def __init__(self, num_classes=100,  # Common medical conditions
                 knowledge_dim=512,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.encoder = MultiModalEncoder(device)

        # Knowledge base integration (RAG)
        self.knowledge_attention = nn.MultiheadAttention(
            embed_dim=512, num_heads=8, batch_first=True)

        # Diagnostic reasoning layers
        self.reasoning = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.LayerNorm(512)
        )

        # Condition prediction
        self.diagnosis_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Severity estimation
        self.severity_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Treatment recommendation
        self.treatment_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 50)  # Common treatments
        )

    def forward(self, image=None, video=None, text=None, knowledge=None):
        # Encode patient data
        patient_embedding = self.encoder(image=image, video=video, text=text)

        # Integrate medical knowledge (RAG)
        if knowledge is not None:
            # Knowledge shape: (batch, knowledge_items, embedding_dim)
            query = patient_embedding.unsqueeze(1)
            knowledge_enhanced, _ = self.knowledge_attention(
                query, knowledge, knowledge)
            patient_embedding = patient_embedding + \
                knowledge_enhanced.squeeze(1)

        # Diagnostic reasoning
        reasoned = self.reasoning(patient_embedding)

        # Generate outputs
        diagnosis = self.diagnosis_head(reasoned)
        severity = self.severity_head(reasoned)
        treatment = self.treatment_head(reasoned)

        return {
            'diagnosis': diagnosis,
            'severity': severity,
            'treatment': treatment,
            'embedding': patient_embedding
        }


# Alias for compatibility
EmptyBrainAI = DiagnosticModel
