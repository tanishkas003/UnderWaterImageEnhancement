from .white_balance import WhiteBalance
from .clahe import CLAHEEnhancer
from .gamma import GammaCorrection
from .sharpen import Sharpen

class EnhancementPipeline:

    def __init__(self, gamma=1.2, clip_limit=3.0):
        self.white_balance = WhiteBalance()
        self.clahe = CLAHEEnhancer(clip_limit)
        self.gamma = GammaCorrection(gamma)
        self.sharpen = Sharpen()

    def process(self, image):
        image = self.white_balance.apply(image)
        image = self.clahe.apply(image)
        image = self.gamma.apply(image)
        image = self.sharpen.apply(image)
        return image
