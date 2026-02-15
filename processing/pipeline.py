from .white_balance import WhiteBalance
from .clahe import CLAHEEnhancer
from .gamma import GammaCorrection
from .sharpen import Sharpen

from .wcid import WCID
from .dcp import DCP
from .dct import DCT

class EnhancementPipeline:

    def __init__(self, gamma=1.2, clip_limit=3.0):

        self.white_balance = WhiteBalance()
        self.clahe = CLAHEEnhancer(clip_limit)
        self.gamma = GammaCorrection(gamma)
        self.sharpen = Sharpen()

        self.wcid = WCID()
        self.dcp = DCP()
        self.dct = DCT()

    def process(self, image, mode="standard"):

        if mode == "standard":
            image = self.white_balance.apply(image)
            image = self.clahe.apply(image)
            image = self.gamma.apply(image)
            image = self.sharpen.apply(image)

        elif mode == "wcid":
            image = self.wcid.apply(image)

        elif mode == "dcp":
            image = self.dcp.apply(image)

        elif mode == "dct":
            image = self.dct.apply(image)

        return image
