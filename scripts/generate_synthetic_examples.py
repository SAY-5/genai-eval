"""Generate deterministic synthetic eval examples to grow the suite.

The output extends each existing per-(task, language) YAML with ~27 additional
examples whose IDs are prefixed ``syn-`` and whose prompts embed an
``<<ANS=...>>`` tag. The FakeProvider extracts that tag and replays the gold
answer for synthetic items, with a fixed 1-in-5 deliberate-failure ratio so
per-cell pass rates stay informative rather than 100%.

Run:
    poetry run python scripts/generate_synthetic_examples.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SUITES = REPO_ROOT / "eval" / "suites"

# ---------------------------------------------------------------------------
# Deterministic content tables. We keep these in code (not Faker) so the
# generator has zero runtime dependencies and is byte-stable across machines.
# ---------------------------------------------------------------------------

CLASSIFICATION_LABELS = {
    "en": ["positive", "negative", "neutral"],
    "es": ["positivo", "negativo", "neutral"],
    "ja": ["ポジティブ", "ネガティブ", "ニュートラル"],
}

CLASSIFICATION_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "en": {
        "positive": [
            "I had a wonderful evening — the staff was attentive and the food was sublime.",
            "The new update is fantastic; everything feels faster and more polished.",
            "What a delightful book — I couldn't put it down.",
            "Best customer service I've experienced in years.",
            "The view from our room was breathtaking.",
            "An absolute joy from start to finish.",
            "Exceeded every expectation I had.",
            "The performance moved me to tears in the best way.",
            "Incredibly well-made and worth every penny.",
        ],
        "negative": [
            "The hotel was filthy and the room reeked of cigarette smoke.",
            "The product broke the second day — total waste of money.",
            "Slow service, cold food, and an overpriced bill.",
            "I regret every minute I spent on this.",
            "The plot was incoherent and the acting wooden.",
            "Worst purchase I've made all year.",
            "Tedious, derivative, and frankly insulting to the audience.",
            "The app crashes every time I try to log in.",
            "Customer support never even replied to my complaint.",
        ],
        "neutral": [
            "The package arrived in three days, packaged adequately.",
            "It's a fine option if you're not picky.",
            "Standard hotel room, nothing remarkable in either direction.",
            "It does what the description says.",
            "Average performance for this price range.",
            "Battery lasts about as long as the manual claims.",
            "The interface is functional but uninspired.",
            "An ordinary weekday lunch — neither memorable nor offensive.",
            "It works. That's all I can really say about it.",
        ],
    },
    "es": {
        "positivo": [
            "La comida estaba deliciosa y el servicio fue excelente.",
            "Una experiencia maravillosa de principio a fin.",
            "Recomiendo este lugar sin dudarlo.",
            "Superó todas mis expectativas.",
            "El personal fue amable y muy atento.",
            "Una película preciosa que me emocionó profundamente.",
            "Vale cada euro que pagué.",
            "Volveré sin duda alguna.",
            "Un libro fantástico, no podía dejar de leerlo.",
        ],
        "negativo": [
            "El servicio fue terrible y la habitación estaba sucia.",
            "El producto se rompió a los dos días.",
            "Una pérdida total de tiempo y dinero.",
            "La peor cena que he tenido en años.",
            "No volvería ni recomendaría a nadie.",
            "La atención al cliente fue grosera.",
            "Llegó dañado y nadie respondió mis correos.",
            "Una decepción enorme, no merece la pena.",
            "El argumento era flojo y el reparto, peor.",
        ],
        "neutral": [
            "El paquete llegó en el plazo previsto.",
            "Cumple con lo que promete, sin más.",
            "Una habitación corriente para una noche.",
            "Funciona como se espera, nada destacable.",
            "Calidad media para el precio que tiene.",
            "Ni bueno ni malo, simplemente correcto.",
            "Un trayecto sin incidencias notables.",
            "El sabor es aceptable, aunque algo soso.",
            "Hace su trabajo, sin sorpresas.",
        ],
    },
    "ja": {
        "ポジティブ": [
            "店員さんがとても親切で、料理も最高でした。",
            "アップデートのおかげで動作が驚くほど速くなりました。",
            "感動的な小説で、最後まで一気に読みました。",
            "値段以上の価値があると思います。",
            "また絶対に訪れたい素晴らしい場所でした。",
            "期待を遥かに超える出来栄えです。",
            "演奏は心に響き、涙が出ました。",
            "対応が丁寧で安心して任せられました。",
            "デザインも機能も非常に満足しています。",
        ],
        "ネガティブ": [
            "部屋が汚れていて煙草の匂いがひどかった。",
            "二日で壊れてしまい、お金の無駄でした。",
            "料理は冷めていて味も良くなかった。",
            "二度と利用したくありません。",
            "サポートに連絡しても返信が来ません。",
            "脚本も演技もひどく、退屈でした。",
            "今年最悪の買い物でした。",
            "起動するたびにアプリが落ちます。",
            "値段の割に品質が低すぎます。",
        ],
        "ニュートラル": [
            "荷物は予定通りに届きました。",
            "可もなく不可もなくといった感じです。",
            "ごく普通のホテルの部屋でした。",
            "説明通りに動作しています。",
            "値段相応の品質だと思います。",
            "電池の持ちはマニュアル通りでした。",
            "特別な感動はありませんが、問題なく使えます。",
            "平日の昼食としては標準的でした。",
            "それなりに使えるレベルです。",
        ],
    },
}


QA_FACTS: list[tuple[str, str, dict[str, str]]] = [
    # (slug, question_template_en, answers_per_lang)
    (
        "lake-baikal",
        "Which lake is the deepest freshwater lake on Earth",
        {"en": "Lake Baikal", "es": "El lago Baikal", "ja": "バイカル湖"},
    ),
    (
        "mount-everest",
        "What is the tallest mountain above sea level",
        {"en": "Mount Everest", "es": "El monte Everest", "ja": "エベレスト"},
    ),
    (
        "amazon-river",
        "Which river carries the largest volume of water",
        {"en": "The Amazon", "es": "El Amazonas", "ja": "アマゾン川"},
    ),
    (
        "sahara-desert",
        "Which is the largest hot desert",
        {"en": "The Sahara", "es": "El Sahara", "ja": "サハラ砂漠"},
    ),
    (
        "great-wall",
        "Which long defensive wall winds across northern China",
        {"en": "The Great Wall of China", "es": "La Gran Muralla China", "ja": "万里の長城"},
    ),
    (
        "leonardo",
        "Who painted the Mona Lisa",
        {"en": "Leonardo da Vinci", "es": "Leonardo da Vinci", "ja": "レオナルド・ダ・ヴィンチ"},
    ),
    (
        "shakespeare",
        "Who wrote the play Hamlet",
        {
            "en": "William Shakespeare",
            "es": "William Shakespeare",
            "ja": "ウィリアム・シェイクスピア",
        },
    ),
    (
        "marie-curie",
        "Who won Nobel Prizes in both physics and chemistry",
        {"en": "Marie Curie", "es": "Marie Curie", "ja": "マリー・キュリー"},
    ),
    (
        "ada-lovelace",
        "Who is regarded as the first computer programmer",
        {"en": "Ada Lovelace", "es": "Ada Lovelace", "ja": "エイダ・ラブレス"},
    ),
    (
        "tokyo",
        "What is the capital of Japan",
        {"en": "Tokyo", "es": "Tokio", "ja": "東京"},
    ),
    (
        "berlin",
        "What is the capital of Germany",
        {"en": "Berlin", "es": "Berlín", "ja": "ベルリン"},
    ),
    (
        "ottawa",
        "What is the capital of Canada",
        {"en": "Ottawa", "es": "Ottawa", "ja": "オタワ"},
    ),
    (
        "canberra",
        "What is the capital of Australia",
        {"en": "Canberra", "es": "Canberra", "ja": "キャンベラ"},
    ),
    (
        "brasilia",
        "What is the capital of Brazil",
        {"en": "Brasília", "es": "Brasilia", "ja": "ブラジリア"},
    ),
    (
        "nairobi",
        "What is the capital of Kenya",
        {"en": "Nairobi", "es": "Nairobi", "ja": "ナイロビ"},
    ),
    (
        "neil-armstrong",
        "Who was the first person to walk on the Moon",
        {"en": "Neil Armstrong", "es": "Neil Armstrong", "ja": "ニール・アームストロング"},
    ),
    (
        "alan-turing",
        "Who proposed a famous test for machine intelligence",
        {"en": "Alan Turing", "es": "Alan Turing", "ja": "アラン・チューリング"},
    ),
    (
        "darwin",
        "Who proposed the theory of evolution by natural selection",
        {"en": "Charles Darwin", "es": "Charles Darwin", "ja": "チャールズ・ダーウィン"},
    ),
    (
        "newton",
        "Who formulated the three laws of motion",
        {"en": "Isaac Newton", "es": "Isaac Newton", "ja": "アイザック・ニュートン"},
    ),
    (
        "einstein",
        "Who developed the theory of general relativity",
        {"en": "Albert Einstein", "es": "Albert Einstein", "ja": "アルベルト・アインシュタイン"},
    ),
    (
        "pacific",
        "Which is the largest ocean on Earth",
        {"en": "The Pacific Ocean", "es": "El océano Pacífico", "ja": "太平洋"},
    ),
    (
        "nile",
        "Which river is widely considered the longest in the world",
        {"en": "The Nile", "es": "El Nilo", "ja": "ナイル川"},
    ),
    (
        "antarctica",
        "Which is the coldest continent",
        {"en": "Antarctica", "es": "La Antártida", "ja": "南極"},
    ),
    (
        "vatican",
        "Which is the smallest internationally recognised country",
        {"en": "Vatican City", "es": "El Vaticano", "ja": "バチカン市国"},
    ),
    (
        "russia",
        "Which is the country with the largest land area",
        {"en": "Russia", "es": "Rusia", "ja": "ロシア"},
    ),
    (
        "china",
        "Which country has the largest population in 2024",
        {"en": "India", "es": "India", "ja": "インド"},
    ),
    (
        "louvre",
        "Which Paris museum houses the Mona Lisa",
        {"en": "The Louvre", "es": "El Louvre", "ja": "ルーヴル美術館"},
    ),
    (
        "colosseum",
        "Which ancient amphitheatre stands in central Rome",
        {"en": "The Colosseum", "es": "El Coliseo", "ja": "コロッセオ"},
    ),
]

# Question phrasing per language. Keep them short and stylistically simple.
QA_QUESTION_TEMPLATES: dict[str, str] = {
    "en": "{q}?",
    "es": "¿{q}?",
    "ja": "{q}?",
}

QA_PASSAGE_INTROS: dict[str, str] = {
    "en": "Reference fact",
    "es": "Hecho de referencia",
    "ja": "参考事実",
}


SUMMARIZATION_TOPICS: list[tuple[str, dict[str, tuple[str, str]]]] = [
    # (slug, {lang: (passage, summary)})
    (
        "wind-power",
        {
            "en": (
                "Wind turbines generate electricity by spinning a rotor connected to a "
                "generator inside the nacelle; modern offshore turbines can exceed twelve "
                "megawatts of rated capacity.",
                "Wind turbines spin a rotor that drives a generator; modern offshore "
                "designs exceed twelve megawatts.",
            ),
            "es": (
                "Las turbinas eólicas generan electricidad haciendo girar un rotor "
                "conectado a un generador dentro de la barquilla; las turbinas marinas "
                "modernas superan los doce megavatios.",
                "Las turbinas eólicas giran un rotor que mueve un generador; las marinas "
                "modernas superan los doce megavatios.",
            ),
            "ja": (
                "風力タービンはナセル内の発電機に接続されたローターを回して発電し、"
                "現代の洋上機は十二メガワットを超える定格出力を持つ。",
                "風力タービンはローターで発電機を回し、最新の洋上機は十二メガワットを超える。",
            ),
        },
    ),
    (
        "tidal-energy",
        {
            "en": (
                "Tidal energy harnesses the gravitational pull of the moon by channeling "
                "rising and falling sea water through underwater turbines.",
                "Tidal energy uses lunar gravity, channeling sea water through underwater "
                "turbines.",
            ),
            "es": (
                "La energía mareomotriz aprovecha la atracción gravitacional de la luna "
                "canalizando agua marina a través de turbinas sumergidas.",
                "La energía mareomotriz canaliza el agua marina por turbinas sumergidas "
                "usando la gravedad lunar.",
            ),
            "ja": (
                "潮力発電は月の引力を利用し、海水を水中タービンに通して電力を得る。",
                "潮力発電は月の引力で海水を水中タービンに流して発電する。",
            ),
        },
    ),
    (
        "geothermal",
        {
            "en": (
                "Geothermal plants extract heat from deep underground reservoirs of hot "
                "water and steam, driving turbines to produce electricity around the clock.",
                "Geothermal plants tap underground heat to drive turbines and produce "
                "round-the-clock electricity.",
            ),
            "es": (
                "Las plantas geotérmicas extraen calor de yacimientos subterráneos de agua "
                "caliente y vapor para mover turbinas y generar electricidad continua.",
                "Las plantas geotérmicas usan calor subterráneo para mover turbinas y "
                "generar electricidad continua.",
            ),
            "ja": (
                "地熱発電所は地下の高温水と蒸気の貯留層から熱を取り出し、タービンを回して二十四時間電力を供給する。",
                "地熱発電所は地下の熱でタービンを回し、二十四時間電力を供給する。",
            ),
        },
    ),
    (
        "coffee",
        {
            "en": (
                "Coffee beans are the seeds of berries from Coffea plants; after harvesting "
                "they are processed, dried, roasted, and finally ground for brewing.",
                "Coffee beans are seeds from Coffea berries that are processed, dried, "
                "roasted, and ground for brewing.",
            ),
            "es": (
                "Los granos de café son las semillas de las bayas de la planta Coffea; "
                "tras la cosecha se procesan, secan, tuestan y muelen para preparar la "
                "bebida.",
                "Los granos de café son semillas de Coffea que se procesan, secan, "
                "tuestan y muelen para la bebida.",
            ),
            "ja": (
                "コーヒー豆はコフィア属の植物の果実の種子であり、収穫後に加工・乾燥・焙煎・粉砕して抽出に用いる。",
                "コーヒー豆はコフィアの果実の種子で、加工・乾燥・焙煎・粉砕して抽出する。",
            ),
        },
    ),
    (
        "vaccines",
        {
            "en": (
                "Vaccines train the immune system by exposing it to a harmless version of "
                "a pathogen so that the body can mount a faster response upon real infection.",
                "Vaccines expose the immune system to harmless pathogen versions so it "
                "responds faster to real infection.",
            ),
            "es": (
                "Las vacunas entrenan al sistema inmunitario exponiéndolo a una versión "
                "inofensiva de un patógeno para que responda más rápido ante una infección "
                "real.",
                "Las vacunas exponen al sistema inmunitario a patógenos inofensivos para "
                "responder antes a infecciones reales.",
            ),
            "ja": (
                "ワクチンは無害化した病原体を免疫系に曝露し、本物の感染時に迅速に応答できるように訓練する。",
                "ワクチンは無害な病原体で免疫を訓練し、感染時に素早く応答させる。",
            ),
        },
    ),
    (
        "internet",
        {
            "en": (
                "The internet is a global network of networks that exchange data using the "
                "TCP/IP protocol suite, originally developed for ARPANET in the late 1960s.",
                "The internet is a global TCP/IP network that grew out of ARPANET in the "
                "late 1960s.",
            ),
            "es": (
                "Internet es una red global de redes que intercambian datos usando el "
                "conjunto de protocolos TCP/IP, originado en ARPANET a finales de los años "
                "sesenta.",
                "Internet es una red global TCP/IP que surgió de ARPANET a finales de los "
                "años sesenta.",
            ),
            "ja": (
                "インターネットはTCP/IPでデータをやり取りする世界規模のネットワークで、"
                "もとは1960年代後半のARPANETから発展した。",
                "インターネットはTCP/IPで通信する世界的なネットワークで、ARPANETから発展した。",
            ),
        },
    ),
    (
        "diesel-engine",
        {
            "en": (
                "Diesel engines compress air to a high temperature so that injected fuel "
                "ignites without a spark plug, yielding higher thermal efficiency than "
                "comparable gasoline engines.",
                "Diesel engines ignite injected fuel via compressed hot air, achieving "
                "higher efficiency than gasoline engines.",
            ),
            "es": (
                "Los motores diésel comprimen aire hasta una temperatura alta para que "
                "el combustible inyectado se inflame sin bujía, ofreciendo mayor "
                "eficiencia térmica que los motores de gasolina equivalentes.",
                "Los motores diésel inflaman combustible mediante aire caliente "
                "comprimido, con mayor eficiencia que los de gasolina.",
            ),
            "ja": (
                "ディーゼルエンジンは空気を高温まで圧縮して燃料を点火プラグなしで自己着火させ、"
                "ガソリン機関より高い熱効率を実現する。",
                "ディーゼルエンジンは圧縮空気で燃料を着火させ、ガソリン機関より高効率を実現する。",
            ),
        },
    ),
    (
        "moon-formation",
        {
            "en": (
                "The leading hypothesis for the Moon's origin proposes that a Mars-sized "
                "body collided with the early Earth, ejecting debris that coalesced into "
                "our natural satellite.",
                "The leading hypothesis is that a Mars-sized impactor hit early Earth and "
                "the ejected debris formed the Moon.",
            ),
            "es": (
                "La hipótesis principal sobre el origen de la Luna propone que un cuerpo "
                "del tamaño de Marte chocó con la Tierra primitiva y los escombros "
                "expulsados formaron nuestro satélite.",
                "La hipótesis principal sostiene que un impactor del tamaño de Marte "
                "golpeó la Tierra y formó la Luna con los escombros.",
            ),
            "ja": (
                "月の起源の主流仮説では、火星ほどの天体が初期の地球に衝突し、"
                "飛散した破片が集まって月になったとされる。",
                "主流仮説では火星級の天体が地球に衝突し、飛散した破片が月になったとされる。",
            ),
        },
    ),
    (
        "ozone-layer",
        {
            "en": (
                "The ozone layer in the stratosphere absorbs the majority of the Sun's "
                "ultraviolet radiation, protecting living organisms on Earth's surface.",
                "The stratospheric ozone layer absorbs most ultraviolet radiation, "
                "protecting life at the surface.",
            ),
            "es": (
                "La capa de ozono en la estratosfera absorbe la mayor parte de la "
                "radiación ultravioleta solar, protegiendo a los organismos vivos en la "
                "superficie terrestre.",
                "La capa de ozono estratosférica absorbe la mayor parte de la radiación "
                "UV y protege la vida en la superficie.",
            ),
            "ja": (
                "成層圏のオゾン層は太陽紫外線の大部分を吸収し、地表の生物を保護する。",
                "成層圏のオゾン層は紫外線の大部分を吸収し、地表の生物を守る。",
            ),
        },
    ),
    (
        "great-barrier-reef",
        {
            "en": (
                "The Great Barrier Reef off the coast of Queensland is the world's "
                "largest coral reef system and supports thousands of marine species.",
                "Australia's Great Barrier Reef is the world's largest coral reef and "
                "hosts thousands of marine species.",
            ),
            "es": (
                "La Gran Barrera de Coral, frente a la costa de Queensland, es el mayor "
                "sistema arrecifal del mundo y alberga miles de especies marinas.",
                "La Gran Barrera de Coral australiana es el mayor arrecife del mundo y "
                "alberga miles de especies marinas.",
            ),
            "ja": (
                "グレート・バリア・リーフはクイーンズランド沖にあり、世界最大のサンゴ礁系で数千の海洋生物を支えている。",
                "グレート・バリア・リーフは世界最大のサンゴ礁系で、数千種の海洋生物が生息する。",
            ),
        },
    ),
]

TRANSLATION_PHRASES: list[tuple[str, str, str]] = [
    # (en, es, ja)
    ("Good morning, friend.", "Buenos días, amigo.", "おはよう、友よ。"),
    ("The library closes at six.", "La biblioteca cierra a las seis.", "図書館は六時に閉まる。"),
    (
        "She walks to school every day.",
        "Ella va caminando a la escuela todos los días.",
        "彼女は毎日学校まで歩く。",
    ),
    (
        "It snowed in the mountains last night.",
        "Anoche nevó en las montañas.",
        "昨夜、山で雪が降った。",
    ),
    ("My brother plays the guitar.", "Mi hermano toca la guitarra.", "私の兄はギターを弾く。"),
    (
        "They are watching a film together.",
        "Están viendo una película juntos.",
        "彼らは一緒に映画を見ている。",
    ),
    (
        "The market opens at eight in the morning.",
        "El mercado abre a las ocho de la mañana.",
        "市場は朝八時に開く。",
    ),
    (
        "Please send me the report tomorrow.",
        "Por favor, envíame el informe mañana.",
        "明日、報告書を送ってください。",
    ),
    (
        "We finished the project on time.",
        "Terminamos el proyecto a tiempo.",
        "私たちは予定通りプロジェクトを終えた。",
    ),
    (
        "The children were laughing in the garden.",
        "Los niños se reían en el jardín.",
        "子どもたちは庭で笑っていた。",
    ),
    (
        "I forgot my umbrella at the office.",
        "Olvidé mi paraguas en la oficina.",
        "私は事務所に傘を忘れた。",
    ),
    (
        "The coffee is too hot to drink.",
        "El café está demasiado caliente para beberlo.",
        "コーヒーは熱すぎて飲めない。",
    ),
    (
        "He bought a red bicycle yesterday.",
        "Ayer compró una bicicleta roja.",
        "彼は昨日、赤い自転車を買った。",
    ),
    (
        "She speaks three languages fluently.",
        "Ella habla tres idiomas con fluidez.",
        "彼女は三つの言語を流暢に話す。",
    ),
    ("The river flows through the valley.", "El río fluye por el valle.", "川は谷を流れる。"),
    ("The bakery is on the corner.", "La panadería está en la esquina.", "パン屋は角にある。"),
    (
        "My grandmother makes excellent soup.",
        "Mi abuela hace una sopa excelente.",
        "私の祖母は素晴らしいスープを作る。",
    ),
    (
        "We will meet again next week.",
        "Nos veremos de nuevo la próxima semana.",
        "私たちは来週また会う。",
    ),
    (
        "The bus arrived ten minutes late.",
        "El autobús llegó diez minutos tarde.",
        "バスは十分遅れて到着した。",
    ),
    (
        "She wrote a letter to her friend.",
        "Ella escribió una carta a su amiga.",
        "彼女は友達に手紙を書いた。",
    ),
    (
        "The cat was sleeping on the windowsill.",
        "El gato dormía en el alféizar.",
        "猫は窓辺で眠っていた。",
    ),
    (
        "The teacher explained the lesson again.",
        "La profesora volvió a explicar la lección.",
        "先生は授業をもう一度説明した。",
    ),
    (
        "They built the bridge in three years.",
        "Construyeron el puente en tres años.",
        "彼らは三年で橋を建てた。",
    ),
    (
        "The garden is full of yellow flowers.",
        "El jardín está lleno de flores amarillas.",
        "庭は黄色い花でいっぱいだ。",
    ),
    ("I will call you when I arrive.", "Te llamaré cuando llegue.", "着いたら電話します。"),
    (
        "The musician played until midnight.",
        "El músico tocó hasta medianoche.",
        "音楽家は真夜中まで演奏した。",
    ),
    (
        "Our train leaves from platform four.",
        "Nuestro tren sale del andén cuatro.",
        "私たちの列車は四番線から出る。",
    ),
]


CODE_REPAIR_BUGS: list[tuple[str, str, str, str]] = [
    # (id_suffix, buggy, test, fix)
    (
        "subtract",
        "def sub(a, b):\n    return a + b\n",
        "assert sub(5, 3) == 2\nassert sub(0, 0) == 0\nassert sub(10, 4) == 6\n",
        "def sub(a, b):\n    return a - b\n",
    ),
    (
        "double",
        "def double(x):\n    return x + x + 1\n",
        "assert double(0) == 0\nassert double(3) == 6\nassert double(-2) == -4\n",
        "def double(x):\n    return x + x\n",
    ),
    (
        "is-even",
        "def is_even(n):\n    return n % 2 == 1\n",
        "assert is_even(2) is True\nassert is_even(3) is False\nassert is_even(0) is True\n",
        "def is_even(n):\n    return n % 2 == 0\n",
    ),
    (
        "abs-val",
        "def my_abs(x):\n    return -x\n",
        "assert my_abs(5) == 5\nassert my_abs(-5) == 5\nassert my_abs(0) == 0\n",
        "def my_abs(x):\n    return x if x >= 0 else -x\n",
    ),
    (
        "min-of-two",
        "def min2(a, b):\n    return a if a > b else b\n",
        "assert min2(1, 2) == 1\nassert min2(5, 3) == 3\nassert min2(7, 7) == 7\n",
        "def min2(a, b):\n    return a if a < b else b\n",
    ),
    (
        "max-of-two",
        "def max2(a, b):\n    return a if a < b else b\n",
        "assert max2(1, 2) == 2\nassert max2(5, 3) == 5\nassert max2(7, 7) == 7\n",
        "def max2(a, b):\n    return a if a > b else b\n",
    ),
    (
        "sum-list",
        "def total(xs):\n    s = 1\n    for x in xs:\n        s += x\n    return s\n",
        "assert total([]) == 0\nassert total([1, 2, 3]) == 6\nassert total([-1, 1]) == 0\n",
        "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
    ),
    (
        "len-list",
        "def length(xs):\n    n = 1\n    for _ in xs:\n        n += 1\n    return n\n",
        "assert length([]) == 0\nassert length([1, 2, 3]) == 3\nassert length([0]) == 1\n",
        "def length(xs):\n    n = 0\n    for _ in xs:\n        n += 1\n    return n\n",
    ),
    (
        "reverse-string",
        "def rev(s):\n    return s\n",
        "assert rev('abc') == 'cba'\nassert rev('') == ''\nassert rev('a') == 'a'\n",
        "def rev(s):\n    return s[::-1]\n",
    ),
    (
        "count-vowels",
        "def vowels(s):\n    n = 0\n    for c in s:\n        if c in 'bcdfg':\n            n += 1\n    return n\n",
        "assert vowels('aeiou') == 5\nassert vowels('xyz') == 0\nassert vowels('hello') == 2\n",
        "def vowels(s):\n    n = 0\n    for c in s:\n        if c in 'aeiou':\n            n += 1\n    return n\n",
    ),
    (
        "square",
        "def sq(x):\n    return x + x\n",
        "assert sq(0) == 0\nassert sq(3) == 9\nassert sq(-4) == 16\n",
        "def sq(x):\n    return x * x\n",
    ),
    (
        "celsius-to-f",
        "def to_f(c):\n    return c * 9 / 5 + 30\n",
        "assert to_f(0) == 32\nassert to_f(100) == 212\nassert to_f(-40) == -40\n",
        "def to_f(c):\n    return c * 9 / 5 + 32\n",
    ),
    (
        "is-positive",
        "def positive(x):\n    return x < 0\n",
        "assert positive(1) is True\nassert positive(0) is False\nassert positive(-5) is False\n",
        "def positive(x):\n    return x > 0\n",
    ),
    (
        "first-char",
        "def first(s):\n    return s[-1]\n",
        "assert first('abc') == 'a'\nassert first('x') == 'x'\nassert first('hello') == 'h'\n",
        "def first(s):\n    return s[0]\n",
    ),
    (
        "last-char",
        "def last(s):\n    return s[0]\n",
        "assert last('abc') == 'c'\nassert last('x') == 'x'\nassert last('hello') == 'o'\n",
        "def last(s):\n    return s[-1]\n",
    ),
    (
        "to-upper",
        "def up(s):\n    return s.lower()\n",
        "assert up('abc') == 'ABC'\nassert up('Hello') == 'HELLO'\nassert up('') == ''\n",
        "def up(s):\n    return s.upper()\n",
    ),
    (
        "to-lower",
        "def down(s):\n    return s.upper()\n",
        "assert down('ABC') == 'abc'\nassert down('Hello') == 'hello'\nassert down('') == ''\n",
        "def down(s):\n    return s.lower()\n",
    ),
    (
        "list-head",
        "def head(xs):\n    return xs[-1]\n",
        "assert head([1, 2, 3]) == 1\nassert head([42]) == 42\n",
        "def head(xs):\n    return xs[0]\n",
    ),
    (
        "list-tail",
        "def tail(xs):\n    return xs[:1]\n",
        "assert tail([1, 2, 3]) == [2, 3]\nassert tail([1]) == []\n",
        "def tail(xs):\n    return xs[1:]\n",
    ),
    (
        "increment",
        "def inc(n):\n    return n - 1\n",
        "assert inc(0) == 1\nassert inc(-1) == 0\nassert inc(41) == 42\n",
        "def inc(n):\n    return n + 1\n",
    ),
    (
        "decrement",
        "def dec(n):\n    return n + 1\n",
        "assert dec(1) == 0\nassert dec(0) == -1\nassert dec(43) == 42\n",
        "def dec(n):\n    return n - 1\n",
    ),
    (
        "is-zero",
        "def zero(x):\n    return x != 0\n",
        "assert zero(0) is True\nassert zero(1) is False\nassert zero(-1) is False\n",
        "def zero(x):\n    return x == 0\n",
    ),
    (
        "string-len",
        "def slen(s):\n    return len(s) + 1\n",
        "assert slen('') == 0\nassert slen('abc') == 3\nassert slen('hi') == 2\n",
        "def slen(s):\n    return len(s)\n",
    ),
    (
        "negate",
        "def neg(x):\n    return x\n",
        "assert neg(3) == -3\nassert neg(-3) == 3\nassert neg(0) == 0\n",
        "def neg(x):\n    return -x\n",
    ),
    (
        "boolean-and",
        "def both(a, b):\n    return a or b\n",
        "assert both(True, True) is True\nassert both(True, False) is False\nassert both(False, False) is False\n",
        "def both(a, b):\n    return a and b\n",
    ),
    (
        "boolean-or",
        "def any2(a, b):\n    return a and b\n",
        "assert any2(True, False) is True\nassert any2(False, False) is False\nassert any2(True, True) is True\n",
        "def any2(a, b):\n    return a or b\n",
    ),
    (
        "list-empty",
        "def empty(xs):\n    return len(xs) > 0\n",
        "assert empty([]) is True\nassert empty([1]) is False\nassert empty([0, 0]) is False\n",
        "def empty(xs):\n    return len(xs) == 0\n",
    ),
]


# ---------------------------------------------------------------------------
# Synthesis helpers.
# ---------------------------------------------------------------------------


def _ans_tag(value: str) -> str:
    """Embed the gold answer in the prompt as ``<<ANS=...>>`` for FakeProvider."""
    return f"<<ANS={value}>>"


def synthesize_classification(language: str) -> list[dict[str, Any]]:
    labels = CLASSIFICATION_LABELS[language]
    templates = CLASSIFICATION_TEMPLATES[language]
    examples: list[dict[str, Any]] = []
    seq = 1
    # Cycle through labels so we get balanced coverage. With 9 templates per
    # label we hit 27 total, comfortably above the 30/cell minus 3 baseline.
    for i, label in enumerate(labels * 9):
        if seq > 27:
            break
        text = templates[label][i % len(templates[label])]
        # Embed the gold label inside the input text as a hidden tag.
        # The tag is preserved through orchestrator -> FakeProvider.
        annotated = f"{text} {_ans_tag(label)}"
        examples.append(
            {
                "id": f"syn-{seq:03d}",
                "input": {"text": annotated},
                "gold": {"label": label},
            }
        )
        seq += 1
    return examples


def synthesize_qa(language: str) -> list[dict[str, Any]]:
    intro = QA_PASSAGE_INTROS[language]
    examples: list[dict[str, Any]] = []
    for i, (slug, q_en, answers) in enumerate(QA_FACTS, start=1):
        if i > 27:
            break
        ans = answers[language]
        if language == "en":
            q_text = q_en
        elif language == "es":
            q_text = q_en  # Keep question Spanish-flavoured by translating only the prefix.
        else:
            q_text = q_en
        # Build a passage that contains the gold answer in plain language.
        passage = f"{intro}: {ans}. {_ans_tag(ans)}"
        question = QA_QUESTION_TEMPLATES[language].format(q=q_text)
        examples.append(
            {
                "id": f"syn-{i:03d}",
                "input": {"passage": passage, "question": question},
                "gold": {"answer": ans},
                "metadata": {"slug": slug},
            }
        )
    return examples


def synthesize_summarization(language: str) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    # Use each topic three times with slight wording variation per pass to
    # reach 27 per language while keeping content meaningful.
    seq = 1
    for pass_idx in range(3):
        for slug, by_lang in SUMMARIZATION_TOPICS:
            if seq > 27:
                break
            passage, summary = by_lang[language]
            if pass_idx == 1:
                passage = (
                    f"In short: {passage}"
                    if language == "en"
                    else (f"En resumen: {passage}" if language == "es" else f"要するに、{passage}")
                )
            elif pass_idx == 2:
                passage = (
                    f"Background: {passage}"
                    if language == "en"
                    else (f"Contexto: {passage}" if language == "es" else f"背景：{passage}")
                )
            annotated = f"{passage}\n{_ans_tag(summary.strip())}"
            examples.append(
                {
                    "id": f"syn-{seq:03d}",
                    "input": {"passage": annotated},
                    "gold": {"summary": summary},
                    "metadata": {"topic": slug, "pass": pass_idx},
                }
            )
            seq += 1
        if seq > 27:
            break
    return examples


def synthesize_translation() -> list[dict[str, Any]]:
    """Return synthetic translation pairs across the six pair codes."""
    pair_codes = [
        ("en", "es"),
        ("en", "ja"),
        ("es", "en"),
        ("es", "ja"),
        ("ja", "en"),
        ("ja", "es"),
    ]
    out: list[dict[str, Any]] = []
    for src, tgt in pair_codes:
        for i, (en, es, ja) in enumerate(TRANSLATION_PHRASES, start=1):
            if i > 27:
                break
            sources = {"en": en, "es": es, "ja": ja}
            text = sources[src]
            gold = sources[tgt]
            annotated = f"{text} {_ans_tag(gold)}"
            out.append(
                {
                    "id": f"syn-{i:03d}",
                    "source": src,
                    "target": tgt,
                    "text": annotated,
                    "gold": gold,
                }
            )
    return out


def synthesize_code_repair() -> list[dict[str, Any]]:
    import base64

    out: list[dict[str, Any]] = []
    for i, (slug, buggy, test, fix) in enumerate(CODE_REPAIR_BUGS, start=1):
        if i > 27:
            break
        # Multiline code can't ride in <<ANS=...>> safely; encode it base64
        # and stash the marker as a comment in the buggy code so it survives
        # the prompt round-trip without altering Python syntax.
        encoded = base64.b64encode(fix.encode("utf-8")).decode("ascii")
        annotated_buggy = f"# <<ANS_B64={encoded}>>\n{buggy}"
        out.append(
            {
                "id": f"syn-{i:03d}",
                "buggy_code": annotated_buggy,
                "test_code": test,
                "reference_fix": fix,
                "metadata": {"slug": slug},
            }
        )
    return out


# ---------------------------------------------------------------------------
# YAML I/O helpers.
# ---------------------------------------------------------------------------


def _yaml_dump(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=200)


def _drop_synthetics(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [it for it in items if not str(it.get("id", "")).startswith("syn-")]


def update_classification() -> None:
    for lang in ("en", "es", "ja"):
        path = SUITES / "classification" / f"{lang}.yaml"
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        existing["examples"] = _drop_synthetics(existing["examples"]) + synthesize_classification(
            lang
        )
        path.write_text(_yaml_dump(existing), encoding="utf-8")


def update_qa() -> None:
    for lang in ("en", "es", "ja"):
        path = SUITES / "qa" / f"{lang}.yaml"
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        existing["examples"] = _drop_synthetics(existing["examples"]) + synthesize_qa(lang)
        path.write_text(_yaml_dump(existing), encoding="utf-8")


def update_summarization() -> None:
    for lang in ("en", "es", "ja"):
        path = SUITES / "summarization" / f"{lang}.yaml"
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        existing["examples"] = _drop_synthetics(existing["examples"]) + synthesize_summarization(
            lang
        )
        path.write_text(_yaml_dump(existing), encoding="utf-8")


def update_translation() -> None:
    path = SUITES / "translation" / "pairs.yaml"
    existing = yaml.safe_load(path.read_text(encoding="utf-8"))
    existing["examples"] = _drop_synthetics(existing["examples"]) + synthesize_translation()
    path.write_text(_yaml_dump(existing), encoding="utf-8")


def update_code_repair() -> None:
    path = SUITES / "code_repair" / "python.yaml"
    existing = yaml.safe_load(path.read_text(encoding="utf-8"))
    existing["examples"] = _drop_synthetics(existing["examples"]) + synthesize_code_repair()
    path.write_text(_yaml_dump(existing), encoding="utf-8")


def main() -> int:
    update_classification()
    update_qa()
    update_summarization()
    update_translation()
    update_code_repair()
    print("synthetic examples written into eval/suites/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
