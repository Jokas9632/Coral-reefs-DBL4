

from coral_python.coral_trainer import CoralTrainer

trainer = CoralTrainer(overrides={
    "model": "models/yolov11s_coral.yaml",
    "data": "config/coral.yaml",
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "device": "cpu"
})

# Train model
trainer.train()

# Validate model
trainer.validate()

# Predict on a new reef image
results = trainer.predict("sample_reef.jpg")
