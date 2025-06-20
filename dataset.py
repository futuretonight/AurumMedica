# dataset.py
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms


class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, qa_csv_path, img_dir):
        # For this example, let's pretend your CSV has columns: 'image_id', 'question', 'answer_label'
        # self.df = pd.read_csv(qa_csv_path)
        # self.img_dir = img_dir

        # --- Dummy data for demonstration ---
        self.questions = [
            "What do you see in this chest x-ray?",
            "Is there evidence of atrial fibrillation in the ECG strip?",
            "Patient reports a persistent cough, what could this be?"
        ] * 10
        self.image_paths = ["dummy.png"] * 30  # A placeholder image
        self.labels = [torch.randint(0, 10, (1,)).item() for _ in range(30)]
        # --- End dummy data ---

        # Standard image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

        # Create a dummy image if it doesn't exist
        if not Image.open("dummy.png").fp:
            Image.new('RGB', (100, 100)).save('dummy.png')

    def __len__(self):
        # return len(self.df)
        return len(self.questions)

    def __getitem__(self, idx):
        # In a real scenario:
        # question = self.df.iloc[idx]['question']
        # label = self.df.iloc[idx]['answer_label']
        # img_path = os.path.join(self.img_dir, self.df.iloc[idx]['image_id'])
        # image = Image.open(img_path).convert("RGB")

        # Dummy data version:
        question = self.questions[idx]
        label = self.labels[idx]
        image = Image.open("dummy.png").convert("RGB")

        image_tensor = self.transform(image)
        return question, image_tensor, torch.tensor(label)
