import torch
import torchvision


class SingleViewScreeningHead(torch.torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.4),
            torch.nn.Sigmoid(),
            torch.nn.Linear(512, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.Sigmoid(),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        return x


class SingleViewDiagnosisHead(torch.torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.LayerNorm(512),
            torch.nn.Dropout(0.2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(512, 256),
            torch.nn.LayerNorm(256),
            torch.nn.Sigmoid(),
            torch.nn.Linear(256, 128),
            torch.nn.LayerNorm(128),
            torch.nn.Sigmoid(),
            torch.nn.Linear(128, 64),
            torch.nn.LayerNorm(64),
            torch.nn.Sigmoid(),
            torch.nn.Linear(64, 5),
            torch.nn.LayerNorm(5),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class MultiViewDiagnosisHead(torch.torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.LayerNorm(1024),
            torch.nn.Dropout(0.2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 512),
            torch.nn.LayerNorm(512),
            torch.nn.Sigmoid(),
            torch.nn.Linear(512, 256),
            torch.nn.LayerNorm(256),
            torch.nn.Sigmoid(),
            torch.nn.Linear(256, 128),
            torch.nn.LayerNorm(128),
            torch.nn.Sigmoid(),
            torch.nn.Linear(128, 64),
            torch.nn.LayerNorm(64),
            torch.nn.Sigmoid(),
            torch.nn.Linear(64, 5),
            torch.nn.LayerNorm(5),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        x = torch.concat([x[0], x[1]], dim=1)
        x = x.squeeze()
        x = self.fc(x)
        return x


class SingleViewScreening:

    def __init__(self, device, model_weight):
        self.device: str = device
        # 加载编码器
        self.encoder: torch.nn.Module = torchvision.models.video.mvit_v2_s()
        self.encoder.head[-1] = torch.nn.Linear(
            self.encoder.head[-1].in_features, 512)
        checkpoint = torch.load(r"model_data/Encoder.pt",
                                map_location="cpu", weights_only=False)
        self.encoder.load_state_dict(checkpoint)
        self.encoder.eval()
        self.encoder.to(device)

        self.decoder = SingleViewScreeningHead()
        weight_data = torch.load(
            model_weight, map_location='cpu', weights_only=False)
        self.decoder.load_state_dict(weight_data["model"])
        self.decoder.to(self.device)
        self.decoder.eval()

    @torch.no_grad()
    def __call__(self, x):
        feature = self.encoder(x.to(self.device))
        predict = self.decoder(feature)
        return predict


class SingleViewDiagnosis:

    def __init__(self, device, model_weight):
        self.device: str = device
        # 加载编码器
        self.encoder: torch.nn.Module = torchvision.models.video.mvit_v2_s()
        self.encoder.head[-1] = torch.nn.Linear(
            self.encoder.head[-1].in_features, 512)
        checkpoint = torch.load(r"model_data/Encoder.pt",
                                map_location="cpu", weights_only=False)
        self.encoder.load_state_dict(checkpoint)
        self.encoder.eval()
        self.encoder.to(device)

        self.decoder = SingleViewDiagnosisHead()
        weight_data = torch.load(
            model_weight, map_location='cpu', weights_only=False)
        self.decoder.load_state_dict(weight_data["model"])
        self.decoder.to(self.device)
        self.decoder.eval()

    @torch.no_grad()
    def __call__(self, x):
        feature = self.encoder(x.to(self.device))
        predict = self.decoder(feature).softmax(-1).squeeze()
        return predict


class MultileViewDiagnosis:

    def __init__(self, device, model_weight):
        self.device: str = device
        # 加载编码器
        self.encoder: torch.nn.Module = torchvision.models.video.mvit_v2_s()
        self.encoder.head[-1] = torch.nn.Linear(
            self.encoder.head[-1].in_features, 512)
        checkpoint = torch.load(r"model_data/Encoder.pt",
                                map_location="cpu", weights_only=False)
        self.encoder.load_state_dict(checkpoint)
        self.encoder.eval()
        self.encoder.to(device)

        self.decoder = MultiViewDiagnosisHead()
        weight_data = torch.load(
            model_weight, map_location='cpu', weights_only=False)
        self.decoder.load_state_dict(weight_data["model"])
        self.decoder.to(self.device)
        self.decoder.eval()

    @torch.no_grad()
    def __call__(self, x):
        x = x.to(self.device)
        feature1 = self.encoder(x[0])
        feature2 = self.encoder(x[1])
        feature = torch.stack([feature1, feature2])
        predict = self.decoder(feature).softmax(-1).squeeze()
        return predict
