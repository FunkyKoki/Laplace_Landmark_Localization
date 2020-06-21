import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, ratio=1, keyPoints=68, dropOutRate=0.3, temperature=1., imageSize=80):
        super(Generator, self).__init__()
        self.featureMaps = [int(64//ratio), int(128//ratio)]
        self.keyPoints = keyPoints
        self.activation = nn.LeakyReLU(0.2)
        self.dropOutRate = dropOutRate
        self.softmax = nn.Softmax(dim=2)
        self.temperature = temperature
        self.imageSize = imageSize
        self.spatialRange = torch.Tensor(list(range(0, self.imageSize)))

        self.blockE1 = nn.Sequential(
            nn.Conv2d(3, self.featureMaps[0], kernel_size=3, stride=1, padding=1),
            self.activation
        )
        self.blockE2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(self.featureMaps[0], self.featureMaps[0], kernel_size=3, stride=1, padding=1),
            self.activation,
            nn.Dropout2d(self.dropOutRate)
        )
        self.blockE3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(self.featureMaps[0], self.featureMaps[0], kernel_size=3, stride=1, padding=1),
            self.activation,
            nn.Dropout2d(self.dropOutRate)
        )
        self.blockE4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(self.featureMaps[0], self.featureMaps[0], kernel_size=3, stride=1, padding=1),
            self.activation,
            nn.Dropout2d(self.dropOutRate)
        )
        self.blockD4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(self.featureMaps[0], self.featureMaps[0], kernel_size=3, stride=1, padding=1),
            self.activation,
            nn.Dropout2d(self.dropOutRate),
            nn.Conv2d(self.featureMaps[0], self.featureMaps[0], kernel_size=1, stride=1, padding=0),
            self.activation,
            nn.Upsample(scale_factor=2)
        )
        self.blockD3 = nn.Sequential(
            nn.Conv2d(self.featureMaps[1], self.featureMaps[0], kernel_size=3, stride=1, padding=1),
            self.activation,
            nn.Dropout2d(self.dropOutRate),
            nn.Conv2d(self.featureMaps[0], self.featureMaps[0], kernel_size=1, stride=1, padding=0),
            self.activation,
            nn.Upsample(scale_factor=2)
        )
        self.blockD2 = nn.Sequential(
            nn.Conv2d(self.featureMaps[1], self.featureMaps[0], kernel_size=3, stride=1, padding=1),
            self.activation,
            nn.Dropout2d(self.dropOutRate),
            nn.Conv2d(self.featureMaps[0], self.featureMaps[0], kernel_size=1, stride=1, padding=0),
            self.activation,
            nn.Upsample(scale_factor=2)
        )
        self.blockD1 = nn.Sequential(
            nn.Conv2d(self.featureMaps[1], self.featureMaps[0], kernel_size=3, stride=1, padding=1),
            self.activation,
            nn.Dropout2d(self.dropOutRate),
            nn.Conv2d(self.featureMaps[0], self.featureMaps[0], kernel_size=1, stride=1, padding=0),
            self.activation,
            nn.Upsample(scale_factor=2)
        )
        self.blockT = nn.Sequential(
            nn.Conv2d(self.featureMaps[1], self.featureMaps[1], kernel_size=5, stride=1, padding=2),
            self.activation,
            nn.Dropout(self.dropOutRate),
            nn.Conv2d(self.featureMaps[1], self.keyPoints, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.keyPoints),
            self.activation,
            nn.Dropout(self.dropOutRate),
            nn.Conv2d(self.keyPoints, self.keyPoints, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.keyPoints),
            self.activation,
            nn.Conv2d(self.keyPoints, self.keyPoints, kernel_size=1, stride=1, padding=0),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        E1 = self.blockE1(x)  # output shape: (b, 64, 80, 80)
        E2 = self.blockE2(E1) # output shape: (b, 64, 40, 40)
        E3 = self.blockE3(E2) # output shape: (b, 64, 20, 20)
        E4 = self.blockE4(E3) # output shape: (b, 64, 10, 10)

        D4 = self.blockD4(E4) # output shape: (b, 64, 10, 10)

        D3 = self.blockD3(torch.cat((D4, E4), dim=1)) # output shape: (b, 128, 20, 20)
        D2 = self.blockD2(torch.cat((D3, E3), dim=1)) # output shape: (b, 128, 40, 40)
        D1 = self.blockD1(torch.cat((D2, E2), dim=1)) # output shape: (b, 128, 80, 80)

        out = self.blockT(torch.cat((D1, E1), dim=1)) # output shape: (b, 68, 80, 80)

        hDistribution = self.softmax(self.temperature*torch.sum(out, dim=3))
        wDistribution = self.softmax(self.temperature*torch.sum(out, dim=2))
        spatialRange = self.spatialRange.type_as(hDistribution)+1
        
        hMean = torch.sum(hDistribution*spatialRange, dim=2, keepdim=True)
        wMean = torch.sum(wDistribution*spatialRange, dim=2, keepdim=True)
        
        hVarianceLaplacian = torch.sum(hDistribution*torch.abs(hMean.repeat(1, 1, self.imageSize)-spatialRange), dim=2, keepdim=True)
        wVarianceLaplacian = torch.sum(wDistribution*torch.abs(wMean.repeat(1, 1, self.imageSize)-spatialRange), dim=2, keepdim=True)

        return torch.cat((wMean, hMean, wVarianceLaplacian, hVarianceLaplacian), dim=2), out

    def calculateLoss(self, predict, target):
        # predict format: batch x keypointNumber x 4(x_mean, y_mean, x_variance, y_variance)  <->  q(x): miu_1; lambda_1
        # target format: batch x keypointNumber x 4(x_mean, y_mean, 1., 1.)  <->  p(x): miu_2; lambda_2
        predict, target = predict.view(-1, 4), target.view(-1, 4)
        
        # for x
        miu_1, lambda_1 = predict[:, 0], predict[:, 2]
        miu_2, lambda_2 = target[:, 0], target[:, 2]
        KLx = self.KLDivergenceBetween2LaplacianDistribution(miu_1, lambda_1, miu_2, lambda_2)

        # for y
        miu_1, lambda_1 = predict[:, 1], predict[:, 3]
        miu_2, lambda_2 = target[:, 1], target[:, 3]
        KLy = self.KLDivergenceBetween2LaplacianDistribution(miu_1, lambda_1, miu_2, lambda_2)

        loss = (KLx+KLy).mean()
        assert ~torch.isnan(loss)
        return loss

    def KLDivergenceBetween2LaplacianDistribution(self, miu_1, lambda_1, miu_2, lambda_2, epsilon=1e-9):

        return torch.log(lambda_2/(lambda_1+epsilon)) - 1 + torch.abs(miu_2 - miu_1)/lambda_2 + lambda_1/lambda_2*torch.exp(-torch.abs(miu_2 - miu_1)/(lambda_1+epsilon))

