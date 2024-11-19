import os
# import boto3
import sys  # Add this line to fix the error
import json
# import tensorflow as tf
from io import BytesIO
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np

from botocore.exceptions import ClientError

from my_lambda_function import lambda_handler

# Create a mock event. Adjust this according to your Lambda function's expected event structure.
event = {
    "rawPath": "/pred",
    "httpMethod": "POST",
    "headers": {
      "Content-Type": "application/json"
    },
    "body": "{\"key\": \"value\"}"
}
event2 = {
    'rawPath': '/pred',
    'body': json.dumps({
        'input': {
            'ImageData': '/9j/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAAUCAEAAQAEABEAAREAAhEAAxEA/8QAGwABAQEBAQEBAQAAAAAAAAAAAAIBAwQFBgf/xAAyEAACAgEEAgEEAQMEAQUBAAAAAQIRIQMSMUEiUWEEEzJxgSNCkQUUobEzQ1JiwfHh/9oADgQAAAEAAgADAAA/APowSrB9eCVHpglR/fzx/WQg036PL9TGLTPN9TGDt+gfB+obhxdXwfI1ZOPHFnytR7apur4BGtrS1NFQJ1dXdpqKJ1NVyhtQPPv8afRy3WqfRy32s9A2LccWZFtYZie3F8gNqrfIulbHAGnJqW3+0Rk1NrpmptOuge7Q0lqSTq0ezSgptN5PTpaanJOsA98Ixjpy2/5PZGMVB1x7PXGMVB1x7B8z6mSy0+zwa7Vtni12tzoHkt8nm5dnDh4Agry/8iMbyYovLBc06VLDNdtY4KknhdA5e/2c+G/VkPkGbm0mkLuqD4QL5jnk6Pg1Aham10v5OcZtPBilkGtNZWGzXd+mHj9g1NxwapOKCbVAZ3c4C3NvOBJO+cAqUkuOGU5JLHAb9cMErUlfGCVqO/gxN9LD7BEpJSjFKryznKS3qKxfJjlToGatRiq5bE6jFU8mSdK0DJLbBZrOTJKorNB4j+wXBxav2dISjttmxa74BEvKV1xg5yW+V1hGS8ndY4BX5K7Kq1do1K1dgiNKJieDE01wC3w2in2zXeaBilFpNfkZcatcmWsPsH6rS+raifoNP6ppH39P6ppfsHi+o+qmm4vs8mt9Q1Jr2eXW13dPsHzdXV3ScTxampbo8c9S5Uwct9qznuvNEWnkE/F4MWf0TXzgGXTr2ZdOjPgDlsLyZvLB006lOKbq/fRUczSeDY5kldA+vppfb8as+lpqP28NJo+hppODpqweZ/UPSclZy++4blZyers3KweDUnuX/J45y3LmzyTldgi/FrslPxMu07B005Vh8FaclVMqMuLBs5/28fJspVjo2cuEDzz8lS4s4S8l/Jxef0Cl4xdPj/grEYl1SwBbpXw3g28Z7CfsFwhc221TNhC5N2shRuQMlFLU3K2qDSWq5coOPk3lguNPyaaVYLTjltM1beaYOes2oVdZOerJqFIjUb28grSSbVsyC4sRXAGuox1Mc4K1ElLBWoqngHF/lZy/u+TlebBbVRTfsrbSVlXeQc5q8romavPolrtf8gKXi9vRil4vaN6rAKg07S4Khm1ZVIBXbguezFduKMjy0Al16LjGlTCVME7pvcsHNyk210Zcm2gbCFKjYxpFKKB+j+o0Xse10z6mte106Po6t1h0DybZqKc43Ryg2oXJWck3VyVg8U4yueptpPhHBp+c2sM4yjJuU6wwcYuo01n2c4vxppL5IWEkwbsbx7N2Pi8DbbroE7UprlUZsqf8Dat36ATyYYnkGpvTTfZV7VZV7bbB20/qXGGW76OkPqKhy7LjrtQ7sHPU1HqNqnlHOcpSb+SJScm8PgHNp/j2jGqVdk8A2aScV38CVY9myXANducUhnckMtoFaqvCK1E3hcDUysdA5/8Ap7WsnJZhXZKzGuweqGlpPSTlLNZPTDT03BOT6yd1CGxNvoHCSzFfODnLpfJzdUgXpxcqTKhFywbFcIFOKjJxZVKM3FmvDaYOM3x6OOo8nOTwCNXc5Ric9RycoojU3XFA6ONRw69nVx8cOi68eQTCW2W6fHtkqVO5PHsyLqVy49sEupNuKpdEp7nfRmG8cAxOTkovjk1XJ01gU7roCeot21J57J1NRYikZKSul3yDY6dRa6kUtKo1eGFBJVeGCYramTDxuwsJ/AMTqX7DdPAXIMtqaoxSe4bvKgXKVK2VKVZYfsHRakHotV5FqcPtNVktyj9uqyD7stR5V91k98pPg97bugRq6ngiZyqCJm6jQPFKalPb8dnJyUntOTkm6BMvp8Y7D0cYMel2DY/TvZufCKjo1Hc+BHSezd0DhqKKfH8nKdeiJrPAOKqU3RxWZHPDkA/JJGtbo0jatYBEW7SSymTHpdp5J7S7QNlJ8ejZSfZrlYNTcpX2bmUrCbbsG8SbdWGs32O2wbPUqq5E57WmjZS28Az7iitzyN9eXKM3dtA5rdqTtYVEU5ytKsEpNuwdX4U7yU3tSzkp4rOQQtV6i4ppmKcpxuuGFJtccA989un9OtRcnsbUNBTSyehuK0lJLIPH9x6snLikeRTeq3Lho4bt7cvQOcnSSVmS/FJEt0ga1uab4Rkluab4RkrdegdXJKGDq5rZjBt+PoHDUae1ZqzhOSdLNESp1zQLqMX4ttF7UktrLpJqngEfcUb9siWpsXyyXNR/YFNu2ao27kjEs2wWtS3XaL+5eL4K3X4giVJu8ENJW/ZLSjyDlpv+o0+jnCVyr0TBvcwdWs7UdGndFpPgFwjFxafLxkuCTTUuTUlTsHPc0tjSOW6ltrsm/wC0H6DcpLJ9fElZ9Z01YInBSVJ8kyhGWE+SJQjLCYJ0fobmpOQ0vo1v3Nmaf0lz3N49A9K0Y7tvo9H2lur0dvtK9qBb0FsdtUdPsrY7eC/srZl4B8r6vQ2Sbt1R876jRcZP0zwa+lsk30Dx1XJ5qz8HCgG7kvVi8oPlUCI+MrWCU6l6JWJAqUVKd9s1pSn+zeZArEezcLljC7BzSW11lPglNbXWV0FhP5Ajl0zE08MAucarGCpdYwGqSQEfx4XyFxlBXWQcncp8OjnTlK6dEyywdIRSl8cnTTVOi44B019RvSVcIvVm/t44RU5eCroHGMdkeeTlGOyLfs5rFv2CqUYqiuIjhcgmUbSvghrcs8Bq1QMS8kuUP7kuifhZQJk93/SMl5L0HT/kEacpU0+CNOT4kTF9MG7fK2sDbulngV7Bc5YtPBc5UuTXJqNA2rdrkqKTyb+uQRKpXFLJE2pPauTH5YQENPZ8NdiMNnQjGmBFu7EW3kRAT/qUE39ygn5UBG98rygr+5K1aC/KSaB96D3wTbo+lpz3wT4PpQluVt0DlqNqkuURKTUsdEyk1LHQOunJNbt2fR6dOScbvJ1g78rz6B6NOST8nk6wkrts6QklK28g2eq446ZUtRr9Mueo1/IPmfVam6bV4PBrzcpNXg8GvO5P0DyUlFr0eWqtM4JYYOX3OEuCN9NJcE7neAbiSKpNmVYLck4dX+ynKOx4ya620gRtbmms1ycWm5Knfsna91rNLINcWv4Om2n+ikrf6Bihm0FDytG1TwC23KSima/KW1Mfk6QIbSlTwY+ab4DSvPQIg22k1WSYybklVE3dLsHWLWc+XR0j3nJqfPsHLUlTpvx7OOpJp112TJv+OwarcFfBVtxXo1N0vQG7C9GXa/RmKBTbaNeVjk2XQJi0m5P9GcW/4MToE6jzirJm8quSZvKoEq0uMkRXtGLGOwFK4Oyk/Bi8MGqDqlj9mNS4M8uFj9g2L2teiotxpIpYayClt3uVBNKe7s1VusDXlFq4/wAla07VxE5egRCSaT5REZ9hS7BrjUm0zXHO6zGnfINhTfybCm/lmxkgfRWptSS6O61NqVM9MZ0gc/8AcOepteKJWs56m0xarlOgd1FxjVuj1xTjHk7K0Dto7pNOTx0zrpbnVs6adtqwej6tOP06d0dvqVt0Uzr9Ta0k+GD5Nb5PPwfO/KR8/wDOXNsHLUjUeTnqRwyJLAONJVm2ji1VZ4ItewXCNRTawWldMqKBjko2sUY5JNxMunXQN0mo2+VybCopvkRe22soB6u7Uxwzd+6V9Gp3L9gnd5NdE7m21fBnbQMbrUUuDG/NMN1JA2abSp2xPKw8iWQZxpX2Y7ULXJn9lg6JRpSv9nVVSlf7KSVJoHNv7iljHshtTtpYJtu1WAT05Proy7i2+hXfoFflFpMzlNJjlNAyH432hDCt5YVJfIMk7MlK/wCSZMGUlK2YvyHYLTW8u1bNunwCJNyV7aRFueUqRl7ldA6KP9NtvLOlLZb5NSuOewcVXRyx0SqqgXJraU8xNbBiqCy+cGLxX/Biwv2AopYRiikalSAlKpP2bJtT6Mk/IBpRjgYSDraD35aysov2drB5le9S4zROm7mmTF+SaB74ar2VJ/5PoQ1KhTZ6o6j2pMHv+mem4VJ0ezQ2OOXR7NFQccugcf8AUtdfbUE7SOX1usnBRTwjn9Zqpw2p8A+TCW1yyfO021Jo+fF02DNWdtuxN85End5yCIvyzwyIvOeGSnYM1bcUk+GNS6VdMyeEgbKnHnJsmnG+zW7VgyKcISJj46bMSqLoGRV/yTEJYB0WmpN3/wDp0jDdKmaldgicVJqPCsjUjckuMmSyqB003GmjpBp4ZcWqyBHR+5q7U+ejft79TbYWmpy2p8gvU+n2ZTKloqC5KlpKOUDzaVyTiumeeHlaWKOMc2gVt3RlF4tlJXFxfZS7T7AVxavliqaTYWOQZNJUZNLDJnWAZHz46MXl/BmH/ACaTaF0xfXQNpbtxtZuzXh2DX+KT4Nf40YninwDm7qSObeGh0wZCNL5MhBpfPomKpZ5B1cdqtl04q2U7StgjUScVXPRkswVchq4r/gCOI3IRxG2YrUcgS/N/rAlmfvHIfIMik+xFJmxpg+jOTjFtI6S8YtnWWOAeNN283eSNNUmznFO+Qa5y4Lc5JUVuawDotecIUpZsv704wpMpa01HDBqm9RNN38lKTmss3e5LLB505SnSwjim5SpYOabbwDdXxSXZupjHYnaQKVVTx8hK1TNWV6Aapc4DwvgPCy8AjTuTbZkcttmRy2C5NtcY6Km7jhYNbf8AyLS6yyY0v5MXP7Bu6TkpMrdK7KvysEu5T/+yZXKfyTLMsA2VJ38GvFP4ya8LIOmnqf3I6Qnas2E+0wTrar1cN4J1JvUtPg2c97p8AiMdkrt0RGO2V9USopNu8AN1KVctmN0212S7t12CpR31WWslSjuSrLRrVpJcoEN7sd2c36fNh5pdgqMHGODpHTqNoza6wDnNpNJMibVpGSdJegVKo5qxLxDaTzkE7tyMT3GYbwCnGkr5K28XyVVAPbGVqrNdRa4sVTvAJ3ObzxZDm5WmY3uwwJKlfAWFfAapegL5fIw02Zhv2B0sG2kkxXoGx8m/SNhcm6WCl5A9etqUkly+BqSeEuy5yqq7B54xXDds2KSw+SEv8gSV/wJrdhGuwF+LC/FpBcMHR/gmsezq0vtp3RTzFVgEJqKtYr/AIIjSWDOI4BDblO2S7lO36My3bfQKbp5NumOHyCW21glu1gPKBWnBuaiu+jUnuUUuTYpuSQOuppzjSbqJ0lCUaT4KnFrDdIHNyW5JEOS3JIzcrxyDNSaTRk5UTJqwZHEvh9hP/8ATVyDXLy4Ku3xgN+QMXi8/wAEJqLf7wZiP/0Do9NLTujq9OoWU44T6AhG1tWWjI/+1GpYpZBya83Frvk5NeTXyQ8yqgRGclr4baMhOX3eeCVJqeHYL2u7fJri9zk0a7tt8gKbS5tGqbXeApP3gHKWZL0n2cp3KS9Ih5ddL2Do6avtlySaRrr1kDTUeBBKzYJIGzxPP4roqVxl/wDH0bLD+Ac1b1Hea4Iy5u80TefYOkcQd8t4Liltd89FJYt8giN6jSIVzJT3UDWtr2hrbg1qgbhX2vRuM4wZwDNW4q06+ENRuKuLqzZWladA9cYqUm5ZX/R00knJ7lZcVbdrAOMVcpWZFXJkpeTQJym75TCuN3zZqxd9AuMX3wy1FPns1Lv2Ck8bayuCuI7az0bwqBG1rk57ZLklrNAWrq89mtrcjXyDm7WpysnOWNS75JdKV2DptS5LS5Nq/gG6Ors+olNq0kbp6m3VlJ5KhPbqOXIOmrr/AHIqP82XPU3RS/kqepvSX8g5KUW6vLOe6P8ALITXHbBiSk3f9vFmRW5vd1wIpW75QEpJQVcX2U3tgbdL4sGRW1uTzaMji3zaMqm37Aj5tWlhkpbmr9mJbuQddypx6OzaraX1XQIWpKCt9nNScUYpOKtUDpHQUtN+XyXHSuDz8mqFp2wcJOMZJri6fycpbYyTXDeSJVFquOweqouDquD1JR2P9FtKmweHyU66PC8Sro4vpdAtq5N8ouk3g15d9AxYjb4MXDszgGx21fs2NU2wuPgCSdNuhNNxZrUqdgzSXvvgzTWMmQXvsDmW1dFXb2ropO8LoGxjU6fIUanT5MSqVMEa9265ROu/XJmo+a5BaalBL2apJwSrkRzFKgTOm12l2TNW0+UjXX+AenT3Rg7XJ2hFxjnsuKrD7BPGpgxYnSG2pIDCnkr+52bxK2BJttpvAavvgPPHAKUqzwUmlTbo1fIOepJ7bTpE6sm47lwTL2v8grYq3NcmKNrc+xXdAxNODaSRkacW0jbTjwCHNtUqviydzfHPBCb+AZHFpLJMVTa7NXNAuL7OiwrsJXkEbE5rJKgnK7MpNgqdX8lzcUr7Nl7aBOpHC9EaibiqwGsIHWLTis8f8Fwql8FJJrIIxFOS7ZLpZ9szCzxkFJW91lxTb3G1eQTN2lXESJO1a6Jbws4QNi5KObyapPar7NTxbvIImnKLg32RNbltRklaaBu96MNvsrc9PTqjN21VQJlF3b7Ocou7fZNPkFRWLsqPFmqgFtlBqgqlFqhakmqwDlF7fHv0Qnt8eyVjHYLptZKe7srPYK01tjTeVwIxpVYSrAOc24NSS5Mm3BqSXJjbjlIG7nKSbNvdJMy3KVgpx3JtGtXdGpXbAhp7Ek/QjClT9GpUgXPa48UnguVbaDpr94B2Sba/4OqydVbeUDNj5aM2Plo1ppK0CHFJq+SGvIn9gVz+wuGGgTebV10Y23ldcGW3mwJzSjTyZKdRpmSlSpoBOTlGV4XQTbcXeEMtp3hAamU0nlo2bw6eWU+Mdg5wTULeGuyYqoW+SIrFvHyDI25J9EwzJMRu0wW5JP8AZcpK/hs1vywCsJm1RuF+wGt0jJLcx+Tp4BWtNfbXFLkvUlH7X65Nk6SrrkE/TaUtRfHRz+nhKa5wZpRlP9AzUWPhM3UjhfDMmv8AgHSKqCadlx/BOyo8cg5xUop4u2RFOLeOyUq47YNbxS5ZspJYXZt4pAxRaWesmJNfxkyq/gEyg9SSb4RMk5tN9GTTlkFN1LPr/Br5WRVNMEVLa/XwZT2t9E26foGab8KfZMcqnwxF2qYNpb93Lujajv3UGrmmkCtSbapGzm2lSNk2+AIpyVlRW5X6Ectu+AJNSjT64EqlGn0G7Tb6BElsiqIa2xTRLdR4BWlaV1h+zdO4q+mbG0r6Bst0VvbNbpbgm/yYEWpqnwIvcs8Gp2D7ENBVG1k+lpaK2xTWT6ENLCTqwPqdLbptpZN+o09unaQ19PbDCyD5lNu2eCrZ46zYMnHw5/gSj4B3QIinXwSrqujM/wAA1tXS74MbW7CEvQOkY1pt457LjFKBqrb1gHByTni69nG03jghO5Y4A1VuhebN1HcUJW4g3Tjuj+jdNWhBWgbGG6MnSpdmpJqWOBttSfoGt3G3hjc2rao1u1fAJqVJrtinhrsxXiuwbNbYOs2JR2p1mypKk2s2CtLVlpOlwxp6j0pUlhmxm4ccMG6iuHpMuauN3yJK1nAG6tPkbqgZwgRbWmtq5ItqPiZ+KVdgR/De+TEqW98hfju7Bqnh2sspSx8s3djIJVvCWLIWX8Gc/oBtNuH/ACbi3Ex48QbH8GujY/g10I8P0DhaUluycXVq8kYTyDq5JY9lulg2/QNirdpZKgm3dZNjd2DIycdRpvxoyLcZtPgyLqTBk2rtISa6Emnwgam5cjc266NTtAyb2rLJm9qyHhZYOsNGc9LKuPKOsNKU9P45KjpyceMcghQqf/xRMYVLPBNeWVgH6nS0U6b4Pv6endN8H29LTt2wc9SP3N0UsGakd6aSwZqR32kgfO1Po3Dhc5PBqfTuPXJ45/TuDB5dTSa57OThtWezlKLXIODe3Bxn4rGDlJ1wDi21qt9JHFtqbb4Ibz+gdYyqDu6fJalS7p8mp4/YJVRW6qTJVRV1hmXtjhYBkmtmeGG1tzhM1vx/YEWqpeuQnhJegvSBcW43Hqi4/wDt6N4Tj0DdKDnqfBsIuU6NjG2CtVbW1ihqJxbV4Eltbygc4NuaS7Mg3KSiSm20gd9SK2cc9nTUgkuC5VVA46ialtXHZylako9ENNYsFpKUf/spU1T4Kw4gmNRbpY6CqLxwjFy8YBE6yvZE/wDsyS6Aw69djDr0MP8AQNaVOnTFKnmg6axgEar2puK8mZq4Vx5ZOo3FWgTBzSzgmG5LyZMW0gV9p6isr7e5Kitu5A1x81foONTX6Ma8qXoFXtyv/wBLb25RV7QRNOTzwznO5P4Idt5BqjKeIrApyxFYNSbdJYBsYbW7KjGnTKUaBktNOLbzQlFNZMcVecguOvOOko8lLVktNRK+41BIGPlKmTT4ZDvckD9f9MrVM/TaCvDP0ehnDwDrLThHJ0cIxz0dHCEfLkHzvqJR3tp5PDrNbmzw6zW60+QfP1YqUXfvg8epHdHg8s1a+QeHWhmujya0d1JHm1EugcprwtdIjUX9N0RJXEHNJ7lFvDZyVqSTeGSul7B21Y7Vmq6O2pFLmi5KlVA5JNwWP4OdbllYJWY5B0UUttlKNV6NWKBcqu0i5NPornhAlSrKkTGTStPJibu7Byk3LUnbdMiTlLUabwS25Skm+QXBVTOmmttM2KSoFakpThTu76Go3NV2bK5LPNgadbLeWsGaa8c8oyNVwBOTUaiJOlSNbqNJA5uahjmyXLY/dmXQJcJTe54XohxlN7n/AIMcZNNvgHSC2qjpGttUal40CZOThmq9kztw5wTJvZkExbT2vleyVOnXonnAKUd7vii0t7vhIpLc74oFwck8cFxtPDwUnTpcAiW7dl5Ocr3Nku7tsCOXSCe7gLINdfi+DXV0zZcIHTQ1'
        }
    })
}

# Call the lambda function directly
response = lambda_handler(event2, None)

# Print the response
print(response)


