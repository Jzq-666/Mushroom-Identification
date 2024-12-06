#!/usr/bin/env python
# pylint: disable=unused-argument
# This program is dedicated to the public domain under the CC0 license.

import logging
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms
from torch import nn


# 定义 ShuffleNet2 模型
class ShuffleNet2(nn.Module):
    def __init__(self, num_classes, input_size=224, net_type=2, dropout_rate=0.5):
        super(ShuffleNet2, self).__init__()
        assert input_size % 32 == 0

        self.stage_repeat_num = [6, 10, 6]
        if net_type == 0.5:
            self.out_channels = [3, 24, 48, 96, 192, 1024]
        elif net_type == 1:
            self.out_channels = [3, 24, 116, 232, 464, 1024]
        elif net_type == 1.5:
            self.out_channels = [3, 24, 176, 352, 704, 1024]
        elif net_type == 2:
            self.out_channels = [3, 24, 244, 488, 976, 2048]
        else:
            raise ValueError("net_type must be one of 0.5, 1, 1.5, or 2")

        self.conv1 = nn.Conv2d(3, self.out_channels[1], 3, 2, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        in_channels = self.out_channels[1]

        self.stages = []
        for stage_idx in range(len(self.stage_repeat_num)):
            out_channels = self.out_channels[2 + stage_idx]
            repeat_num = self.stage_repeat_num[stage_idx]
            for i in range(repeat_num):
                downsample = (i == 0)
                self.stages.append(ShuffleBlock(in_channels, out_channels, downsample=downsample))
                in_channels = out_channels
        self.stages = nn.Sequential(*self.stages)

        in_channels = self.out_channels[-2]
        self.extra_conv = conv_bn(in_channels, in_channels * 2, stride=1)
        in_channels *= 2
        out_channels = self.out_channels[-1]

        self.conv5 = conv_1x1_bn(in_channels, out_channels, 1)
        self.g_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stages(x)
        x = self.extra_conv(x)
        x = self.conv5(x)
        x = self.g_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# 定义通道混洗和卷积函数
def channel_shuffle(x, groups=2):
    batch_size, channels, width, height = x.shape
    group_channels = channels // groups
    x = x.view(batch_size, groups, group_channels, width, height)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, width, height)
    return x


def conv_1x1_bn(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def conv_bn(in_channels, out_channels, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class ShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ShuffleBlock, self).__init__()
        self.downsample = downsample
        half_channels = out_channels // 2
        if downsample:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 2, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, half_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_channels),
                nn.ReLU(inplace=True)
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, half_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(half_channels, half_channels, 3, 2, 1, groups=half_channels, bias=False),
                nn.BatchNorm2d(half_channels),
                nn.Conv2d(half_channels, half_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_channels),
                nn.ReLU(inplace=True)
            )
        else:
            assert in_channels == out_channels
            self.branch2 = nn.Sequential(
                nn.Conv2d(half_channels, half_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(half_channels, half_channels, 3, 1, 1, groups=half_channels, bias=False),
                nn.BatchNorm2d(half_channels),
                nn.Conv2d(half_channels, half_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.downsample:
            out = torch.cat((self.branch1(x), self.branch2(x)), 1)
        else:
            channels = x.shape[1]
            half = channels // 2
            x1 = x[:, :half, :, :]
            x2 = x[:, half:, :, :]
            out = torch.cat((x1, self.branch2(x2)), 1)
        return channel_shuffle(out, 2)

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

GENDER, PHOTO, LOCATION, BIO, CONTINUE, CLARIFY, FINAL_QUESTION = range(7)

# Load your trained model here
def load_model(model_path, num_classes):
    model = ShuffleNet2(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Image preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Import necessary libraries
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters
)
from io import BytesIO
from PIL import Image
import torch

# Define constants for conversation states
PHOTO, CLARIFY, FINAL_QUESTION, CONTINUE = range(4)

# Mushroom classes with toxicity information
classes = {
    '雲芝': 'non-toxic', '冬菇': 'non-toxic', '冬蟲夏草': 'non-toxic',
    '出血齒菌': 'toxic', '變綠紅菇': 'toxic', '大青褶傘': 'toxic',
    '大鹿花菌': 'non-toxic', '寬鱗多孔菌': 'non-toxic', '尾花籠頭菌': 'toxic',
    '乾巴菌': 'non-toxic', '惡魔雪茄蘑菇': 'toxic', '杏鮑菇': 'non-toxic',
    '歐洲黑木耳': 'non-toxic', '毒絲蓋傘': 'toxic', '毒蕈': 'toxic',
    '毒蠅傘': 'toxic', '毛頭鬼傘': 'toxic', '靈芝': 'non-toxic',
    '牛舌菌': 'non-toxic', '狹頭小菇': 'non-toxic', '猴頭菇': 'non-toxic',
    '硫黃菌': 'toxic', '竹蓀': 'non-toxic', '粉紅枝瑚菌': 'non-toxic',
    '糞生黑蛋巢菌': 'non-toxic', '紫蠟蘑': 'non-toxic',
    '綠蓋粉孢牛肝菌': 'toxic', '紅籠頭菌': 'toxic',
    '紅紫柄小菇': 'non-toxic', '細褐鱗蘑菇': 'non-toxic',
    '羊肚菌': 'non-toxic', '美味牛肝菌': 'non-toxic',
    '藍綠乳菇': 'toxic', '裂褶菌': 'toxic',
    '赭紅擬口蘑': 'non-toxic', '金黃鵝膏菌': 'toxic',
    '鱗柄白鵝膏': 'toxic', '鹿蕊': 'non-toxic',
    '黃裙竹蓀': 'non-toxic', '黑松露': 'non-toxic'
}
# Load the model
model_path = "./mm.pth"
num_classes = len(classes)  # Number of classes
model = load_model(model_path, num_classes)
model.to(device)
# Encyclopedia content for mushroom species

encyclopedia = {
    '雲芝': "雲芝（Trametes versicolor）是一種常見的藥用真菌，具有多種生物活性。雲芝常被稱為多彩真菌，其菌蓋呈現彩虹條紋結構，顏色從白色到棕色不等。它在傳統中醫中被用於增強免疫系統，現代科學證實其提取物能促進T細胞活性，用於輔助癌癥治療。此外，雲芝還是森林中的重要分解者，有助於生態平衡。",
    '冬菇': "冬菇（Lentinula edodes）是一種廣泛食用的蘑菇，特別流行於亞洲料理。冬菇含有豐富的維生素D、鉀和抗氧化劑，對免疫系統和骨骼健康有益。它也被認為可以降低膽固醇，改善心血管功能。",
    '冬蟲夏草': "冬蟲夏草（Cordyceps sinensis）是一種珍貴的藥用真菌，主要生長於高原地區。其活性成分蟲草素具有抗腫瘤、抗炎和免疫調節的效果，在傳統中醫中被用於增強體力和治療呼吸系統疾病。",
    '出血齒菌': "出血齒菌（Hydnellum peckii）是一種有毒的蘑菇，以其鮮紅色乳液為特徵。它主要生長於北半球的針葉林中，不適合食用，但在生態學中具有重要作用，是土壤養分循環的一部分。",
    '變綠紅菇': "變綠紅菇（Russula virescens）是一種有毒的蘑菇，菌蓋通常呈現綠色或紅色，外觀引人註目。盡管其顏色美麗，但食用可能導致嚴重的腸胃不適，甚至中毒反應。由於其毒性，需謹慎辨識，最好在專業人士的指導下進行采集。",
    '大青褶傘': "大青褶傘（Lactarius indigo）是一種藍綠色的食用蘑菇，富含維生素和抗氧化劑。它的獨特風味使其在北美和亞洲的傳統湯品中常見。大青褶傘的肉質厚實，適合多種烹飪方式，特別是燉煮，能夠為菜肴增添鮮香。",
    '大鹿花菌': "大鹿花菌（Morchella esculenta）以其蜂巢狀的菌蓋而聞名，是一種珍貴的野生食用菌。它富含蛋白質和多種維生素，常用於高檔餐廳的菜肴中。其獨特的風味和香氣使其成為許多廚師的首選，尤其適合搭配肉類或用於調味。",
    '寬鱗多孔菌': "寬鱗多孔菌（Polyporus squamosus）通常附著於死樹或腐木上，是重要的木材分解者。它具有一定的藥用價值，富含纖維素，適合用於湯品或藥材加工。這種菌類的獨特風味和質感使其在烹飪中也逐漸受到關註。",
    '尾花籠頭菌': "尾花籠頭菌（Mycena haematopus）是一種小型有毒的蘑菇，因其乳紅色液體而得名。主要生長在腐朽的木材上，盡管其外觀美麗，但食用可能導致中毒。這種蘑菇在生態系統中起到分解木質素的作用，促進了營養循環。",
    '乾巴菌': "乾巴菌（Ganoderma applanatum）是一種藥用真菌，其提取物被認為具有抗炎、抗腫瘤和免疫增強的作用。乾巴菌在亞洲傳統醫學中被廣泛使用。",
    '惡魔雪茄蘑菇': "惡魔雪茄蘑菇（Chorioactis geaster）是一種非常罕見的真菌，外形酷似打開的雪茄。主要分布在美國和日本，因其獨特的外觀和生物特性而受到科學研究者的關註。它在生態系統中可能扮演重要角色，盡管食用價值有限。",
    '杏鮑菇': "杏鮑菇（Pleurotus eryngii）是一種受歡迎的食用蘑菇，肉質厚實，富含膳食纖維和蛋白質。常用於亞洲料理，如燉菜和炒菜。其獨特的口感和香氣，使其成為素食者和肉食者的佳選，適合各種烹飪方式。",
    '歐洲黑木耳': "歐洲黑木耳（Auricularia auricula-judae）是一種食用菌，主要用於湯品和沙拉。它含有豐富的膠原蛋白，有助於改善血液循環。黑木耳的獨特口感和營養價值，使其在亞洲料理中廣受歡迎，尤其是在冬季。",
    '毒絲蓋傘': "毒絲蓋傘（Inocybe patouillardii）是一種有毒的蘑菇，含有毒蕈堿，食用可能導致嚴重的中毒反應甚至死亡。由於其毒性，必須小心辨識，避免誤食。此種蘑菇通常生長在潮濕的環境中，與其他植物共生。",
    '毒蕈': "毒蕈（Amanita phalloides）是最致命的蘑菇之一，其毒性來源於α-鵝膏蕈堿。食用可能引起肝功能衰竭，是世界上毒性最強的真菌之一。由於其外觀與可食用蘑菇相似，常導致誤食，需特別小心。",
    '毒蠅傘': "毒蠅傘（Amanita muscaria）以其紅色菌蓋和白色斑點著稱，常見於北半球的針葉林中。儘管具有神經毒性，但在某些文化中被用作宗教儀式的一部分。",
    '毛頭鬼傘': "毛頭鬼傘（Cortinarius orellanus）是一種有毒的蘑菇，含有致腎毒素，可能引發嚴重的腎損傷，需極度小心避免接觸。此種蘑菇常生長在潮濕的森林環境中，其外觀特征可能與可食用蘑菇相似，需謹慎辨識。。",
    '靈芝': "靈芝（Ganoderma lucidum）是中醫中的重要藥材，被視為健康長壽的象征。靈芝富含三萜類和多糖，具有抗炎、免疫調節和抗腫瘤的作用。其在傳統醫學中的使用歷史悠久，現代研究也在不斷驗證其健康益處。",
    '牛舌菌': "牛舌菌（Fistulina hepatica）是一種外形酷似牛舌的食用菌，常見於橡樹上。它富含維生素C和抗氧化劑，對健康有多重益處。其獨特的外觀和味道使其在料理中備受青睞，尤其是在燒烤和燉煮時。",
    '狹頭小菇': "是一種小型的森林真菌，通常生長在腐木上。其顏色鮮艷，具有一定的觀賞價值，雖然食用價值較低，但在自然環境中具有重要的生態作用。",
    '猴頭菇': "猴頭菇（Hericium erinaceus）以其獨特外形聞名，研究表明，猴頭菇可能有助於神經再生，對改善記憶和認知功能具有潛在益處。這種蘑菇富含蛋白質和其他營養成分，是健康飲食的良好選擇。",
    '硫黃菌': "硫黃菌（Laetiporus sulphureus）以其亮黃色菌蓋得名，被稱為「雞肉蘑菇」。這種食用菌具有濃郁的香氣，富含蛋白質，是素食者的佳品。",
    '竹蓀': "竹蓀（Phallus indusiatus）是一種外形獨特的食用菌，常見於亞洲的潮濕森林中。它被認為具有滋補作用，常用於高級料理中。其獨特的外觀和口感使其在美食界受到歡迎。",
    '粉紅枝瑚菌': "粉紅枝瑚菌（Ramaria botrytis）是一種外形如珊瑚的食用菌，因其獨特的形狀和顏色而備受喜愛。其粉紅色外觀和細膩的口感使其成為受歡迎的野生食材，常用於高檔料理中。粉紅枝瑚菌不僅味道鮮美，且富含多種營養成分，適合用於燉煮和清炒等烹飪方式。由於其外觀獨特，這種蘑菇也常被用作裝飾，增加菜肴的視覺吸引力。",
    '糞生黑蛋巢菌': "糞生黑蛋巢菌（Sphaerobolus stellatus）是一種有趣的真菌，以其能夠以爆裂的方式釋放孢子而得名。這種現象不僅有趣，還顯示了其在生態系統中的重要作用。糞生黑蛋巢菌通常生長在腐爛的有機物上，幫助分解和回收土壤中的營養物質。它的孢子釋放機製對生態平衡至關重要，促進了植物的生長和土壤的肥沃。",
    '紫蠟蘑': "紫蠟蘑（Laccaria amethystina）是一種無毒的蘑菇，以其鮮艷的紫色外觀而著稱。它在森林中是常見的觀賞真菌，盡管其食用價值不高，但其獨特的顏色和形態使其在自然界中頗具吸引力。紫蠟蘑通常生長在濕潤的土壤中，常與其他植物共生。雖然不常食用，但其美麗的外觀吸引了許多真菌愛好者和攝影師。",
    '綠蓋粉孢牛肝菌': "綠蓋粉孢牛肝菌（Boletus viridis）是一種有毒的蘑菇，其綠色菌蓋含有強效毒素，食用後可能導致腸胃不適甚至器官損傷。這種蘑菇通常生長在潮濕的森林地帶，與特定樹種共生。由於其外觀與某些可食用蘑菇相似，容易造成誤食，因此在野外采集時需特別謹慎。了解其特征和生長環境對於避免中毒至關重要。",
    '紅籠頭菌': "紅籠頭菌（Hygrophorus camarophyllus）是一種具有鮮紅色菌蓋的可食用菌，因其引人註目的顏色而受到歡迎。它富含維生素和礦物質，常出現在高檔餐廳的菜單中，尤其在湯品和燉菜中廣泛使用。紅籠頭菌的味道鮮美，口感滑嫩，在烹飪時能夠吸收其他調料的香氣，提升整體菜肴的風味。此外，其營養價值使其成為健康飲食的良好選擇。",
    '紅紫柄小菇': "紅紫柄小菇（Mycena haematopus）是一種小型有毒蘑菇，因其紅色汁液而得名，主要生長在腐木上。雖然外觀美麗，但食用可能導致中毒，需謹慎對待。",
    '細褐鱗蘑菇': "細褐鱗蘑菇（Stropharia rugosoannulata）是一種可食用的蘑菇，其菌蓋呈褐色且有細鱗片。它在草地和森林中廣泛分布，適合用於各種菜肴，特別是搭配肉類。",
    '羊肚菌': "羊肚菌（Morchella esculenta）是一種高價值的野生食用菌，具有濃郁的香氣和細膩的口感，經常用於高級烹飪。",
    '美味牛肝菌': "美味牛肝菌（Boletus edulis）是一種廣受歡迎的食用菌，被認為是世界上味道最好的野生蘑菇之一。",
    '藍綠乳菇': "藍綠乳菇（Lactarius indigo）一種外形美麗的可食用菌，主要分布於北美地區，尤其是在潮濕的森林環境中。其顯著的藍綠色菌蓋令人印象深刻，顏色來自於天然色素，且在烹調過程中會產生輕微的變化，使其口感更加豐富。這種蘑菇不僅富含營養，還富含抗氧化成分，常被用來製作湯品和炒菜，增加菜肴的視覺吸引力。",
    '裂褶菌': "裂褶菌（Schizophyllum commune）是一種可食用真菌，廣泛分布於熱帶和亞熱帶地區，尤其是森林中。它以其獨特的裂褶狀菌蓋而聞名，通常生長在腐木和樹幹上。裂褶菌在傳統醫學中被廣泛應用，因其被認為具有抗菌和抗炎的特性。此外，它在木材分解中扮演重要角色，幫助維持生態平衡，促進營養循環。",
    '赭紅擬口蘑': "赭紅擬口蘑（Pseudoclitocybe cyathiformis）是一種無毒的蘑菇，因其獨特的外形而受到關註，常被用作觀賞性真菌。其菌蓋呈赭紅色，形狀類似於碗，因此在自然界中極具辨識度。盡管其食用價值相對較低，但在生態系統中，它仍然發揮著重要的分解作用，幫助分解植物殘骸，促進土壤肥沃。",
    '金黃鵝膏菌': "金黃鵝膏菌（Amanita caesarea）是一種美味的食用菌，以其金黃色菌蓋而聞名。它在古羅馬時期被認為是帝王的食材，貴族們常在宴會上享用。金黃鵝膏菌肉質厚實，口感細膩，適合多種烹飪方式，尤其是燉煮和烤製。其營養豐富，含有多種維生素和礦物質，是高檔菜肴的珍貴原料。",
    '鱗柄白鵝膏': "鱗柄白鵝膏（Amanita vaginata）是一種有毒的蘑菇，主要分布於歐洲和亞洲的森林中，常在潮濕的環境中生長。其菌蓋呈灰白色，外觀與一些可食用蘑菇相似，但其毒性會影響神經系統，食用後可能導致中毒。由於其外觀的迷惑性，需特別小心辨識，避免誤食。",
    '鹿蕊': "鹿蕊（Cortinarius cinnamomeus）是一種常見於北半球的真菌，通常生長在潮濕的森林中。其菌蓋呈棕色，表面光滑，具有一定的觀賞價值。鹿蕊在真菌分類學研究中常被用作參照物，因其外形特征和生長環境對研究者具有重要意義。雖然其食用價值不高，但在生態學研究中扮演著重要角色。",
    '黃裙竹蓀': "黃裙竹蓀（Phallus indusiatus）是一種獨特的菌類，以其網狀結構而聞名，外形獨特且極具美感。常見於亞洲的潮濕森林中，黃裙竹蓀在亞洲料理中廣泛應用，尤其是在湯品和燉菜中。其被認為具有補益作用，有助於增強體力和免疫力。",
    '黑松露': "黑松露（Tuber melanosporum）是一種珍貴的地下真菌，被譽為「廚房的鉆石」。它生長在樹根附近，通常與橡樹和榛樹共生，具有濃郁的香氣，深受廚師的青睞。黑松露在高檔料理中常用於增添風味，尤其是在意大利和法國的菜肴中。由於其稀有性和獨特風味，黑松露成為美食界的珍貴原料，價格昂貴，常用於婚宴和其他重要場合的料理中。"
}

# Start function
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Thank u for using our bot! Please give me ONE picture of the mushroom u would like to recognize and I will tell u its species/感謝您使用我們的機器人！請給我一張你想認識的蘑菇的照片，我會告訴你它的種類。\n"

    )
    return PHOTO

# Photo handling and mushroom recognition
async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle photo input and predict the mushroom class."""
    user = update.message.from_user
    photo_file = await update.message.photo[-1].get_file()
    image_data = await photo_file.download_as_bytearray()

    # Recognize mushroom
    recog_result, probability = predict_image(image_data)

    # Handle special cases for clarification
    if recog_result in ['牛舌菌', '靈芝']:
        context.user_data["clarify_type"] = "confusion1"
        context.user_data["options"] = {'1': '牛舌菌', '2': '靈芝'}
    elif recog_result in ['大青褶傘', '鱗柄白鵝膏']:
        context.user_data["clarify_type"] = "confusion2"
        context.user_data["options"] = {'1': '大青褶傘', '2': '鱗柄白鵝膏'}
    elif recog_result in ['毒絲蓋傘', '冬菇']:
        context.user_data["clarify_type"] = "confusion3"
        context.user_data["options"] = {'1': '毒絲蓋傘', '2': '冬菇'}
    else:
        await update.message.reply_text(
            f"The mushroom is likely to be '{recog_result}' with a probability of {probability:.2f}. "
            "Please consult an expert before consuming or selling."
        )
        await show_encyclopedia(update, recog_result)
        return CONTINUE

    context.user_data["probability"] = probability
    await update.message.reply_text(
        f"This mushroom might be '{context.user_data['options']['1']}' or '{context.user_data['options']['2']}'. "
        "If you want to clarify, please enter 'y'."
    )
    return CLARIFY

def predict_image(image_data):
    """Predict the mushroom species from the uploaded image."""
    image = Image.open(BytesIO(image_data)).convert("RGB")
    input_image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_image)
        probabilities = torch.nn.functional.softmax(output, dim=1).squeeze(0)

        top_prob, top_index = probabilities.max(0)
        top_class = list(classes.keys())[top_index]

    return top_class, top_prob.item()

# Clarification questions
# Clarification questions
async def clarify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_response = update.message.text.lower()
    if user_response == 'y':
        clarify_type = context.user_data.get("clarify_type")
        options = context.user_data.get("options")

        if clarify_type == "confusion1":
            await update.message.reply_text(
                "What is the texture of the mushroom?\n"
                "'1' for soft and elastic, with a sweet and sour smell.\n"
                "'2' for hard surface, woody interior, no noticeable elasticity.\n"
                "3' for unsure, go to the next question.\n"
                "'4' for end.\n"
                "蘑菇的质地如何？\n"
                "1. 肉質較軟，富有彈性，有酸甜的氣味。\n"
                "2. 表面堅硬，內部木質化，沒有明顯的彈性。\n"
                "3. 不確定，下一個問題。\n"
                "4. 结束识别 "
            )
        elif clarify_type == "confusion2":
            await update.message.reply_text(
                "Does the mushroom have a scaly cap surface?\n"
                "'1' for Yes.\n"
                "'2' for No.\n"
                "'3' for unsure, go to the next question."
                "'4' for end.\n"
                "這種蘑菇的菌蓋表面是否有鱗片狀結構？\n"
                "1. 是\n"
                "2. 否\n"
                "3. 不確定，下一個問題。\n"
                "4. 结束识别"
            )
        elif clarify_type == "confusion3":
            await update.message.reply_text(
                "What does the mushroom smell like?\n"
                "'1' for no noticeable scent.\n"
                "'2' for a distinct aroma.\n"
                "'3' for switch to the next question.\n"
                "'4' for end.\n"
                "蘑菇聞起來是什麼味道？\n"
                "1. 沒有明顯氣味\n"
                "2. 代表獨特的蘑菇氣味\n"
                "3. 切換到下一個問題\n"
                "4. 代表結束"
            )
        return FINAL_QUESTION
    else:
        await update.message.reply_text("Clarification skipped.")
        return CONTINUE

# Final clarification questions
async def final_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_response = update.message.text
    clarify_type = context.user_data.get("clarify_type")
    options = context.user_data.get("options")

    if clarify_type == "confusion1":
        if user_response == '1':
            await update.message.reply_text(
                "The mushroom is likely to be 'Fistulina hepatica'. However, please note that we DO NOT recommend you to eat or sell any species of mushrooms before consulting an expert.\n\n"
                "這種蘑菇很可能是「牛舌菌」。但是，請注意，在諮詢專家之前，我們不建議您食用或出售任何蘑菇。"

            )
            await show_encyclopedia(update, '牛舌菌')

        elif user_response == '2':
            await update.message.reply_text(
                "The mushroom is likely to be 'Ganoderma lucidum'. However, please note that we DO NOT recommend you to eat or sell any species of mushrooms before consulting an expert.\n\n"
                "這種蘑菇很可能是「靈芝」。但是，請注意，在諮詢專家之前，我們不建議您食用或出售任何蘑菇。"
            )
            await show_encyclopedia(update, '靈芝')
        elif user_response == '3':
            await update.message.reply_text(
                "Switching to the next question.\n"
                "What is the approximate size of the mushroom?\n"
                "'1' for about 10cm.\n"
                "'2' for about 20cm.\n"
                "'3' for unsure， end this identification. \n"
                "'4' for end. \n"
                "蘑菇的大小大概爲多少？\n"
                "1. 10cm左右\n"
                "2. 20cm左右\n"
                "3. 不确定,结束本次识别\n"
                "4. 结束识别. \n"
            )
            return FINAL_QUESTION
        elif user_response == '4':
            await update.message.reply_text(
                "Thank you so much for using our bot! Remember to always consult an expert and save the sample before you eat any kind of mushrooms just in case!\n\n"
                "非常感謝您使用我們的機器人！請記住，在食用任何種類的蘑菇之前，請務必諮詢專家並保存樣品，以防萬一！"
            )
            return ConversationHandler.END

    elif clarify_type == "confusion2":
        if user_response == '1':
            await update.message.reply_text(
                "This mushroom is likely Parasol Mushroom (大青褶伞). However, please note, we do not recommend consuming or selling any mushrooms without consulting an expert.這種蘑菇很可能是「大青褶伞」。但是，請注意，在諮詢專家之前，我們不建議您食用或出售任何蘑菇。"
        )
            await show_encyclopedia(update, options['1'])
        elif user_response == '2':
            await update.message.reply_text(
                "This mushroom is likely Amanita vaginata (鳞柄白鹅膏). However, please note, we do not recommend consuming or selling any mushrooms without consulting an expert.\n"
                "這種蘑菇很可能是「鳞柄白鹅膏」。但是，請注意，在諮詢專家之前，我們不建議您食用或出售任何蘑菇。"
            )
            await show_encyclopedia(update, options['2'])
        elif user_response == '3':
            await update.message.reply_text(
                "Proceeding to the next question. Does the mushroom have a ring around the stem?\n"
                "'1' for Yes.\n"
                "'2' for No.\n"
                "'3' for unsure， end this identification. \n"
                "'4' for end. \n"
                "這種蘑菇的菌柄是否有菌環？\n"
                "1. 是\n"
                "2. 否\n"
                "3. 不確定，结束本次识别\n"
                "4. 结束识别"
            )
            return FINAL_QUESTION
        elif user_response == '4':
            await update.message.reply_text(
                "Thank you so much for using our bot! Remember to always consult an expert and save the sample before you eat any kind of mushrooms just in case!\n\n"
                "非常感謝您使用我們的機器人！請記住，在食用任何種類的蘑菇之前，請務必諮詢專家並保存樣品，以防萬一！"
            )
            return ConversationHandler.END

    elif clarify_type == "confusion3":
        if user_response == '1':
            await update.message.reply_text(
                "This mushroom is likely to be 'toxic ink cap'. This is a kind of poisonous mushroom, do not eat, so as not to endanger your life.\n"
                "這種蘑菇很可能是「毒絲蓋傘」。這是一種有毒蘑菇，請勿食用，以免危及您的生命安全"
            )
            await show_encyclopedia(update, options['1'])
        elif user_response == '2':
            await update.message.reply_text(
                "The mushroom is likely to be 'shiitake mushroom'. However, please notice that we do NOT recommend u to eat or sell any spieces of  mushrooms before consulting an expert.\n"
                "這種蘑菇很可能是「冬菇」。但是，請注意，在諮詢專家之前，我們不建議您食用或出售任何蘑菇。"
            )
            await show_encyclopedia(update, options['2'])
        elif user_response == '3':
            await update.message.reply_text(
                "Proceeding to the next question. Where did u find the mushroom?\n"
                "'1' for grasslands, wooldland\n"
                "'2'for decayed wood/wooden substrates\n"
                "'3' for unsure， end this identification. \n"
                "'4' for end. \n"
                "你在哪裡找到蘑菇的？\n"
                "1. 代表草原、林地\n"
                "2. 代表於腐爛的木材/木質基材\n"
                "3. 不確定，结束本次识别\n"
                "4. 结束识别"
            )
            return FINAL_QUESTION
        elif user_response == '4':
            await update.message.reply_text(
                "Thank you so much for using our bot! Remember to always consult an expert and save the sample before you eat any kind of mushrooms just in case!\n\n"
                "非常感謝您使用我們的機器人！請記住，在食用任何種類的蘑菇之前，請務必諮詢專家並保存樣品，以防萬一！"
            )
            return ConversationHandler.END
    return CONTINUE


# Show encyclopedia entry for a specific mushroom
# Show encyclopedia entry for a specific mushroom
async def show_encyclopedia(update: Update, mushroom_name: str) -> None:
    """Display the encyclopedia entry for a mushroom."""
    if mushroom_name in encyclopedia:
        entry = encyclopedia[mushroom_name]
    else:
        entry = "No information available for this mushroom. Please consult an expert for accurate identification."
    await update.message.reply_text(
        f"Encyclopedia entry for '{mushroom_name}':\n\n{entry}"
    )


# Continue recognition
async def continue_recognition(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Ask if the user wants to recognize another mushroom."""
    await update.message.reply_text(
        "Do you want to recognize another mushroom?\n1. Continue\n2. End."
    )
    return CONTINUE  # 返回状态以等待用户的响应

async def handle_response(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle the user's response to continue or end the recognition."""
    user_response = update.message.text

    if user_response == '1':
        return await start(update, context)  # Restart the recognition process
    elif user_response == '2':
        await update.message.reply_text(
            "Thank you for using our bot! Remember to always consult an expert before consuming mushrooms."
        )
        return ConversationHandler.END
    else:
        await update.message.reply_text(
            "Please enter '1' to continue or '2' to end."
        )
        return CONTINUE
# Main function
def main():
    """Run the bot application."""
    application = Application.builder().token("7398600993:AAGiLjM7HQD5NhG7b6a4B5r0KwsbORcmW3g").build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            PHOTO: [MessageHandler(filters.PHOTO, photo)],
            CLARIFY: [MessageHandler(filters.TEXT & ~filters.COMMAND, clarify)],
            FINAL_QUESTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, final_question)],
            CONTINUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, continue_recognition, handle_response)],
        },
        fallbacks=[
            CommandHandler("cancel", lambda update, context: update.message.reply_text("Session ended."))],
    )

    application.add_handler(conv_handler)
    application.run_polling()

if __name__ == "__main__":
    main()