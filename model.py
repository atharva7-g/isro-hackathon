class SimpleVideoDataset(Dataset):
    """Simple dataset for video frame sequences"""
    def __init__(self, video_dir, sequence_length=5):
        self.video_dir = video_dir
        self.sequence_length = sequence_length
        self.sequences = self._find_sequences()
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def _find_sequences(self):
        """Find all valid frame sequences"""
        sequences = []
        
        # Assuming frames are named sequentially: frame_001.jpg, frame_002.jpg, etc.
        for video_folder in os.listdir(self.video_dir):
            video_path = os.path.join(self.video_dir, video_folder)
            if not os.path.isdir(video_path):
                continue
                
            frames = sorted([f for f in os.listdir(video_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            # Create sequences of 5 consecutive frames
            for i in range(len(frames) - self.sequence_length + 1):
                sequence = [os.path.join(video_path, frames[i+j]) 
                          for j in range(self.sequence_length)]
                sequences.append(sequence)
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_paths = self.sequences[idx]
        
        # Load frames
        frames = []
        for path in sequence_paths:
            frame = Image.open(path).convert('RGB')
            frames.append(self.transform(frame))
        
        frames = torch.stack(frames)
        
        # Return first 4 as input, last as target
        return frames[:4], frames[4]