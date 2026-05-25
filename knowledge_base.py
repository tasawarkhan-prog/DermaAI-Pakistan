# knowledge_base.py
# Curated skin disease knowledge base with Urdu translations
# Sources: DermNet NZ (public), WHO guidelines, Pakistan DRAP formulary

SKIN_KNOWLEDGE = {
    "acne": {
        "name_en": "Acne Vulgaris",
        "name_ur": "مہاسے",
        "severity": "Low",
        "severity_ur": "کم",
        "description_en": (
            "Acne vulgaris is a chronic inflammatory skin condition affecting hair follicles "
            "and sebaceous glands. It commonly presents as whiteheads, blackheads, papules, "
            "pustules, nodules, or cysts, mainly on the face, chest, and back. "
            "It is caused by excess sebum production, C. acnes bacteria, follicular "
            "hyperkeratinization, and inflammation. Hormonal changes are a major trigger."
        ),
        "description_ur": (
            "مہاسے جلد کی ایک عام بیماری ہے جو بالوں کے فولیکلز اور تیل کے غدودوں کو متاثر کرتی ہے۔ "
            "یہ اکثر چہرے، سینے اور پیٹھ پر ظاہر ہوتی ہے۔ ہارمونز، تیل اور بیکٹیریا اس کی اہم وجوہات ہیں۔"
        ),
        "causes_en": [
            "Excess sebum (oil) production by sebaceous glands",
            "Follicular hyperkeratinization (clogged pores)",
            "Cutibacterium acnes bacterial overgrowth",
            "Hormonal fluctuations (androgens, puberty, menstrual cycle)",
            "Certain medications (steroids, lithium)",
            "Diet high in dairy or high-glycemic foods",
        ],
        "symptoms_en": [
            "Whiteheads (closed comedones)",
            "Blackheads (open comedones)",
            "Small red tender bumps (papules)",
            "Pus-filled pustules",
            "Large solid painful lumps (nodules/cysts)",
            "Post-inflammatory hyperpigmentation or scarring",
        ],
        "when_to_see_doctor_en": (
            "Consult a dermatologist if acne is severe, cystic, causes scarring, "
            "or does not respond to over-the-counter treatments after 2-3 months."
        ),
        "when_to_see_doctor_ur": (
            "اگر مہاسے شدید ہوں، داغ چھوڑ رہے ہوں یا 2-3 مہینے تک عام دوائیں کام نہ کریں تو فوری ماہر جلد سے ملیں۔"
        ),
        "keywords": ["acne", "pimples", "blackheads", "whiteheads", "comedones", "sebaceous", "mhase"],
    },

    "eczema": {
        "name_en": "Atopic Dermatitis (Eczema)",
        "name_ur": "ایگزیما (جلد کی خارش)",
        "severity": "Medium",
        "severity_ur": "درمیانہ",
        "description_en": (
            "Atopic dermatitis is a chronic, relapsing inflammatory skin disease characterised "
            "by intense pruritus (itching), dry skin, and eczematous lesions. It often starts "
            "in childhood and is associated with asthma and allergic rhinitis (the atopic triad). "
            "The skin barrier dysfunction allows allergens and irritants to penetrate, triggering "
            "immune responses."
        ),
        "description_ur": (
            "ایگزیما جلد کی ایک دائمی بیماری ہے جس میں شدید خارش، خشکی اور سرخ دھبے ہوتے ہیں۔ "
            "یہ اکثر بچپن میں شروع ہوتی ہے اور دمہ یا الرجی کے ساتھ ہو سکتی ہے۔"
        ),
        "causes_en": [
            "Genetic predisposition (FLG gene mutations affecting skin barrier)",
            "Immune dysregulation (Th2-skewed immune response)",
            "Environmental triggers: dust mites, pet dander, pollen",
            "Irritants: soaps, detergents, synthetic fabrics",
            "Food allergens (eggs, milk, peanuts in children)",
            "Stress and psychological factors",
            "Climate changes and low humidity",
        ],
        "symptoms_en": [
            "Intense itching, worse at night",
            "Dry, sensitive, scaly skin",
            "Red to brownish-gray patches",
            "Small, raised bumps that may weep fluid",
            "Thickened, cracked, or scaly skin",
            "Raw, swollen skin from scratching",
            "Common sites: inner elbow, behind knees, face, neck",
        ],
        "when_to_see_doctor_en": (
            "Seek medical attention if the rash is severe, widespread, infected (weeping, crusted, "
            "warm), or severely affecting sleep and daily activities."
        ),
        "when_to_see_doctor_ur": (
            "اگر خارش بہت شدید ہو، پورے جسم پر پھیل جائے، یا جلد سے پانی نکل رہا ہو تو فوری ڈاکٹر سے ملیں۔"
        ),
        "keywords": ["eczema", "atopic", "dermatitis", "itchy", "dry skin", "rash", "igzima"],
    },

    "psoriasis": {
        "name_en": "Psoriasis",
        "name_ur": "صدف (چنبل)",
        "severity": "Medium",
        "severity_ur": "درمیانہ",
        "description_en": (
            "Psoriasis is a chronic autoimmune condition causing rapid skin cell turnover, "
            "resulting in thick, scaly plaques. The most common form is plaque psoriasis, "
            "presenting as raised, red patches covered with thick, silvery scales. "
            "It can affect joints (psoriatic arthritis) in about 30% of patients."
        ),
        "description_ur": (
            "صدف ایک مناعتی بیماری ہے جس میں جلد کے خلیے تیزی سے بڑھتے ہیں اور موٹی، چاندی جیسی چھال بناتے ہیں۔ "
            "یہ جوڑوں کو بھی متاثر کر سکتی ہے۔"
        ),
        "causes_en": [
            "Autoimmune dysfunction (T-cell overactivation)",
            "Genetic factors (HLA-Cw6 gene)",
            "Triggers: infections (streptococcal throat), stress",
            "Medications: beta-blockers, lithium, antimalarials",
            "Skin injury (Koebner phenomenon)",
            "Smoking and alcohol consumption",
        ],
        "symptoms_en": [
            "Raised, inflamed, red patches (plaques)",
            "Thick, silvery-white scales on patches",
            "Dry, cracked skin that may bleed",
            "Itching, burning, or soreness",
            "Thickened, pitted, or ridged nails",
            "Swollen and stiff joints (psoriatic arthritis)",
        ],
        "when_to_see_doctor_en": (
            "Always consult a dermatologist. Psoriasis requires prescription treatment. "
            "Seek urgent care if extensive (>10% body), involving joints, or causing severe distress."
        ),
        "when_to_see_doctor_ur": (
            "صدف کے لیے ہمیشہ ماہر جلد سے ملنا ضروری ہے۔ اگر جوڑوں میں درد ہو تو فوری علاج کروائیں۔"
        ),
        "keywords": ["psoriasis", "plaque", "silvery scales", "scaly patches", "sadaf", "chambel"],
    },

    "fungal_infection": {
        "name_en": "Fungal Skin Infection (Tinea / Ringworm)",
        "name_ur": "فنگل انفیکشن (داد)",
        "severity": "Low",
        "severity_ur": "کم",
        "description_en": (
            "Tinea (ringworm) refers to dermatophyte fungal infections of the skin. "
            "Despite the name, no worm is involved. Subtypes include tinea corporis (body), "
            "tinea pedis (athlete's foot), tinea cruris (jock itch), and tinea capitis (scalp). "
            "It is highly contagious and thrives in warm, moist environments."
        ),
        "description_ur": (
            "داد ایک فنگل (پھپھوندی) بیماری ہے جو گرم اور نم ماحول میں پھیلتی ہے۔ "
            "یہ جسم کے مختلف حصوں پر ہو سکتی ہے اور چھونے سے پھیل سکتی ہے۔"
        ),
        "causes_en": [
            "Dermatophyte fungi: Trichophyton, Microsporum, Epidermophyton species",
            "Direct contact with infected person, animal, or soil",
            "Sharing personal items (towels, combs, shoes)",
            "Warm, humid environments (sweating, tight clothing)",
            "Weakened immune system",
            "Poor hygiene and prolonged skin moisture",
        ],
        "symptoms_en": [
            "Ring-shaped, scaly, red patch with clear center",
            "Itching and burning at affected site",
            "Borders may be raised and blistered",
            "Hair loss if scalp is affected (tinea capitis)",
            "Cracking, peeling skin between toes (tinea pedis)",
            "Groin redness with sharp borders (tinea cruris)",
        ],
        "when_to_see_doctor_en": (
            "See a doctor if the infection spreads, involves scalp or nails, "
            "doesn't improve after 2 weeks of antifungal cream, or recurs frequently."
        ),
        "when_to_see_doctor_ur": (
            "اگر داد پھیل رہی ہو، ناخنوں یا سر تک پہنچ جائے، یا 2 ہفتے بعد بھی ٹھیک نہ ہو تو ڈاکٹر سے ملیں۔"
        ),
        "keywords": ["fungal", "ringworm", "tinea", "athlete's foot", "jock itch", "dad", "daad"],
    },

    "scabies": {
        "name_en": "Scabies",
        "name_ur": "خارش (خارش کیڑا)",
        "severity": "Medium",
        "severity_ur": "درمیانہ",
        "description_en": (
            "Scabies is a highly contagious skin infestation caused by the microscopic mite "
            "Sarcoptes scabiei. The mite burrows into the skin to lay eggs, causing intense "
            "itching, especially at night. It spreads through prolonged direct skin contact "
            "and is common in crowded conditions. It is very common in Pakistan."
        ),
        "description_ur": (
            "خارش ایک بہت زیادہ پھیلنے والی بیماری ہے جو ایک چھوٹے کیڑے سے ہوتی ہے جو جلد میں سوراخ کر کے انڈے دیتا ہے۔ "
            "رات کو شدید خارش ہوتی ہے۔ یہ پاکستان میں بہت عام ہے۔"
        ),
        "causes_en": [
            "Sarcoptes scabiei var. hominis mite infestation",
            "Prolonged direct skin-to-skin contact",
            "Shared bedding, clothing, or towels",
            "Crowded living conditions",
            "Weakened immune system (crusted/Norwegian scabies)",
        ],
        "symptoms_en": [
            "Intense itching, severely worse at night",
            "Thin, irregular burrow tracks in skin",
            "Rash with small pimple-like bumps",
            "Common sites: wrists, finger webs, waist, groin, buttocks",
            "Sores from scratching (risk of secondary infection)",
            "Entire household often affected simultaneously",
        ],
        "when_to_see_doctor_en": (
            "Always confirm diagnosis with a doctor. All close contacts must be treated "
            "simultaneously. Seek urgent care for crusted (Norwegian) scabies."
        ),
        "when_to_see_doctor_ur": (
            "ڈاکٹر سے تشخیص ضروری ہے۔ گھر کے تمام افراد کو ایک ساتھ علاج کروانا ضروری ہے۔"
        ),
        "keywords": ["scabies", "itching mite", "sarcoptes", "khaarish", "kharish"],
    },

    "vitiligo": {
        "name_en": "Vitiligo",
        "name_ur": "برص (سفید داغ)",
        "severity": "Low",
        "severity_ur": "کم",
        "description_en": (
            "Vitiligo is a chronic autoimmune condition where melanocytes (pigment-producing cells) "
            "are destroyed, resulting in white patches of skin. It is not contagious or life-threatening "
            "but can cause significant psychological distress. It affects people of all skin types "
            "but is more noticeable in darker skin tones common in Pakistan."
        ),
        "description_ur": (
            "برص ایک مناعتی بیماری ہے جس میں جلد کے رنگ بنانے والے خلیے ختم ہو جاتے ہیں اور سفید دھبے پڑ جاتے ہیں۔ "
            "یہ چھوت کی بیماری نہیں ہے اور خطرناک بھی نہیں ہے۔"
        ),
        "causes_en": [
            "Autoimmune destruction of melanocytes",
            "Genetic predisposition",
            "Triggers: sunburn, skin trauma, stress",
            "Associated with thyroid disease, diabetes, alopecia areata",
        ],
        "symptoms_en": [
            "Milky-white flat patches on skin",
            "Premature whitening of hair on scalp, eyebrows, eyelashes",
            "Loss of color in mucous membranes",
            "Patches may appear anywhere on body",
            "Sun sensitivity in depigmented areas",
        ],
        "when_to_see_doctor_en": (
            "Consult a dermatologist, especially if spreading rapidly. "
            "Treatment can slow progression and restore some pigment."
        ),
        "when_to_see_doctor_ur": (
            "اگر دھبے تیزی سے پھیل رہے ہوں تو ماہر جلد سے ملیں۔ علاج سے رنگ واپس آ سکتا ہے۔"
        ),
        "keywords": ["vitiligo", "white patches", "depigmentation", "baras", "bars", "safed daagh"],
    },

    "contact_dermatitis": {
        "name_en": "Contact Dermatitis",
        "name_ur": "چھونے سے ہونے والی جلد کی سوزش",
        "severity": "Low",
        "severity_ur": "کم",
        "description_en": (
            "Contact dermatitis is skin inflammation caused by direct contact with an irritant "
            "or allergen. Irritant contact dermatitis (ICD) is caused by direct skin damage, "
            "while allergic contact dermatitis (ACD) involves an immune response. "
            "Common triggers in Pakistan include henna, detergents, metals (nickel), "
            "cosmetics, and rubber."
        ),
        "description_ur": (
            "یہ بیماری کسی چیز کے چھونے سے جلد پر سوزش پیدا ہو جاتی ہے۔ "
            "پاکستان میں مہندی، صابن، دھات اور کاسمیٹکس عام وجوہات ہیں۔"
        ),
        "causes_en": [
            "Irritants: soaps, detergents, solvents, bleach",
            "Allergens: nickel (jewellery), rubber, latex",
            "Cosmetics, fragrances, hair dyes (PPD)",
            "Plants: henna, poison ivy",
            "Medications applied to skin",
            "Occupational exposure (healthcare, beauty industry)",
        ],
        "symptoms_en": [
            "Red, itchy rash at contact site",
            "Blisters or bumps that may ooze",
            "Dry, cracked skin",
            "Swelling, burning, or tenderness",
            "Hives (urticaria) in allergic type",
        ],
        "when_to_see_doctor_en": (
            "See a doctor if the rash is widespread, blisters are large, "
            "or symptoms don't improve after avoiding the trigger."
        ),
        "when_to_see_doctor_ur": (
            "اگر دانے بہت پھیل جائیں یا وجہ ہٹانے کے بعد بھی ٹھیک نہ ہوں تو ڈاکٹر سے ملیں۔"
        ),
        "keywords": ["contact", "dermatitis", "rash", "allergy", "irritant", "henna reaction"],
    },

    "urticaria": {
        "name_en": "Urticaria (Hives)",
        "name_ur": "چھپاکی (جھرا)",
        "severity": "Medium",
        "severity_ur": "درمیانہ",
        "description_en": (
            "Urticaria (hives) is a skin reaction causing itchy welts (wheals) that appear suddenly "
            "and typically resolve within 24 hours. It can be acute (< 6 weeks) or chronic (> 6 weeks). "
            "Angioedema (deeper swelling) can accompany hives and may affect the throat — this is a "
            "medical emergency."
        ),
        "description_ur": (
            "چھپاکی میں جلد پر اچانک سرخ، خارش والے دھبے آ جاتے ہیں جو 24 گھنٹوں میں ختم ہو جاتے ہیں۔ "
            "اگر گلے یا زبان پر سوجن ہو تو یہ ایمرجنسی ہے۔"
        ),
        "causes_en": [
            "Food allergies: nuts, shellfish, eggs, milk",
            "Medications: aspirin, NSAIDs, antibiotics (penicillin)",
            "Infections: viral, bacterial",
            "Insect stings or bites",
            "Physical triggers: pressure, cold, heat, exercise",
            "Autoimmune conditions (chronic urticaria)",
            "Latex allergy",
        ],
        "symptoms_en": [
            "Raised, itchy, pink or red welts on skin",
            "Welts vary in size, appear and fade within hours",
            "Burning or stinging sensation",
            "Angioedema: deep swelling around eyes, lips, genitals",
            "Anaphylaxis warning: throat swelling, breathing difficulty",
        ],
        "when_to_see_doctor_en": (
            "Seek EMERGENCY care immediately if there is throat swelling, difficulty breathing, "
            "or dizziness — signs of anaphylaxis. Otherwise, see a doctor for chronic or severe hives."
        ),
        "when_to_see_doctor_ur": (
            "اگر گلے میں سوجن، سانس لینے میں مشکل یا چکر آئیں تو فوری ہسپتال جائیں — یہ ایمرجنسی ہے۔"
        ),
        "keywords": ["hives", "urticaria", "wheals", "welts", "allergic rash", "chhpaki", "jhara"],
    },

    "melanoma": {
        "name_en": "Melanoma",
        "name_ur": "میلانوما (جلد کا سرطان)",
        "severity": "Critical",
        "severity_ur": "نازک",
        "description_en": (
            "Melanoma is the most dangerous form of skin cancer, arising from melanocytes. "
            "Early detection is critical for survival. Use the ABCDE rule to assess moles: "
            "Asymmetry, Border irregularity, Color variation, Diameter > 6mm, Evolution/change. "
            "UV exposure is the primary risk factor."
        ),
        "description_ur": (
            "میلانوما جلد کا سب سے خطرناک سرطان ہے۔ جلدی تشخیص زندگی بچاتی ہے۔ "
            "کسی بھی بدلتے تل کو فوری ڈاکٹر کو دکھائیں۔ سورج کی شعاعیں اس کی بڑی وجہ ہیں۔"
        ),
        "causes_en": [
            "UV radiation (sunlight and tanning beds) — primary risk",
            "Fair skin, light hair and eyes",
            "Family history of melanoma",
            "Many or atypical moles (dysplastic nevi)",
            "Weakened immune system",
            "Previous melanoma or non-melanoma skin cancer",
        ],
        "symptoms_en": [
            "A: Asymmetry — one half unlike the other",
            "B: Border — irregular, ragged, or blurred edges",
            "C: Color — variation (black, brown, red, white, blue)",
            "D: Diameter — larger than 6mm (pencil eraser size)",
            "E: Evolving — any change in size, shape, color, or bleeding",
            "New mole that looks different from others",
        ],
        "when_to_see_doctor_en": (
            "URGENT: See a dermatologist immediately if any mole shows ABCDE changes. "
            "Melanoma is treatable when caught early. Do NOT delay."
        ),
        "when_to_see_doctor_ur": (
            "فوری: کسی بھی بدلتے تل کے لیے ابھی ماہر جلد سے ملیں۔ جلدی علاج میں جان بچانے کا موقع ہے۔"
        ),
        "keywords": ["melanoma", "skin cancer", "mole change", "abcde", "malignant", "melanocyte"],
    },

    "rosacea": {
        "name_en": "Rosacea",
        "name_ur": "گلابی دانے (روزاسیا)",
        "severity": "Low",
        "severity_ur": "کم",
        "description_en": (
            "Rosacea is a chronic inflammatory skin condition causing redness, flushing, "
            "and visible blood vessels mainly on the face. It may also produce acne-like bumps. "
            "It is often confused with acne or sunburn. Triggers include sunlight, hot beverages, "
            "spicy food, alcohol, and stress."
        ),
        "description_ur": (
            "روزاسیا چہرے پر سرخی، جھریاں اور خون کی نظر آنے والی نسیں پیدا کرتا ہے۔ "
            "گرم مشروبات، مصالحے اور دھوپ اسے بڑھاتی ہے۔"
        ),
        "causes_en": [
            "Exact cause unknown; likely combination of genetics and environment",
            "Abnormal immune response and vascular reactivity",
            "Demodex mite overgrowth",
            "Triggers: sun, heat, spicy food, alcohol, exercise, stress",
            "Certain skincare products (alcohol-based)",
        ],
        "symptoms_en": [
            "Persistent facial redness (cheeks, nose, forehead, chin)",
            "Flushing and blushing easily",
            "Visible small blood vessels (telangiectasia)",
            "Swollen bumps, sometimes with pus",
            "Eye irritation (ocular rosacea)",
            "Enlarged nose (rhinophyma) in severe cases",
        ],
        "when_to_see_doctor_en": (
            "See a dermatologist for diagnosis and treatment. Rosacea is manageable but "
            "requires long-term care. Eye involvement (ocular rosacea) needs specialist care."
        ),
        "when_to_see_doctor_ur": (
            "ماہر جلد سے تشخیص کروائیں۔ آنکھوں میں تکلیف ہو تو فوری معائنہ ضروری ہے۔"
        ),
        "keywords": ["rosacea", "facial redness", "flushing", "telangiectasia", "red face"],
    },
}


def get_all_documents():
    """Return all knowledge base entries as a list of text documents for embedding."""
    docs = []
    for condition_key, info in SKIN_KNOWLEDGE.items():
        text = (
            f"Condition: {info['name_en']} ({info['name_ur']})\n"
            f"Description: {info['description_en']}\n"
            f"Causes: {'; '.join(info['causes_en'])}\n"
            f"Symptoms: {'; '.join(info['symptoms_en'])}\n"
            f"When to see doctor: {info['when_to_see_doctor_en']}\n"
            f"Keywords: {', '.join(info['keywords'])}"
        )
        docs.append({
            "key": condition_key,
            "text": text,
            "info": info,
        })
    return docs


def get_condition_info(condition_key):
    """Retrieve full info for a condition key."""
    return SKIN_KNOWLEDGE.get(condition_key, None)
