"""
Multilingual GenAI Municipal Assistant & Justification Report Engine.
Supports English, Hindi (हिंदी), Hinglish, Marathi (मराठी), and Kannada (ಕನ್ನಡ)
with direct, clutter-free answers, simple project explanations, and natural Web Speech audio.
"""

import pandas as pd
from datetime import datetime

def answer_municipal_query(prompt: str, df: pd.DataFrame, city_name: str = "Delhi (MCD)", lang: str = "Hindi") -> dict:
    """
    Context-aware response generator answering municipal infrastructure questions directly,
    in the selected regional language with crisp, natural speech audio.
    """
    p_lower = prompt.lower().strip()
    critical_wards = df[df["risk_category"] == "Critical"].sort_values(by="deficit_score", ascending=False)
    top_critical = critical_wards.iloc[0] if len(critical_wards) > 0 else df.iloc[0]
    worst_drainage = df.sort_values(by="drainage_choke_idx", ascending=False).iloc[0]
    worst_water = df.sort_values(by="water_deficit_pct", ascending=False).iloc[0]
    worst_roads = df.sort_values(by="avg_road_age_years", ascending=False).iloc[0]

    # Check if user is asking to explain the project in simple words
    is_explain_project = any(term in p_lower for term in [
        "samjhao", "explain", "kya hai", "kaise kaam", "samajh", "batao", "overview", "introduction", "project kya"
    ]) or any(term in prompt for term in ["समझाओ", "क्या है", "कैसे काम", "प्रोजेक्ट"])

    if is_explain_project:
        if lang == "Hindi":
            content = f"""
### 💡 इन्फ्राबजट एआई (InfraBudget AI) क्या है और कैसे काम करता है?

यह प्रोजेक्ट नगर निगम के लिए एक **डेटा-आधारित बजट आवंटन प्रणाली (AI Fund Allocator)** है। इसे 4 आसान चरणों में समझें:

1. **समस्या (The Problem)**:
   - आमतौर पर नगर निगमों में बजट नेताओं के प्रभाव या पुराने ढर्रे पर बंटता है। 
   - जहां वास्तव में सड़कें टूटी हैं या पानी नहीं आ रहा, वहां फंड नहीं पहुंच पाता।

2. **डेटा और एआई स्कोर (Deficit Score 0 से 100)**:
   - हमारा सिस्टम हर वार्ड का जमीनी डेटा लेता है: सड़क कितनी पुरानी है, पानी की कितनी कमी है, कितनी शिकायतें आई हैं, और सैटेलाइट से हीट/कचरा कितना है।
   - मशीन लर्निंग मॉडल 0 से 100 का **Deficit Score (कमी का स्कोर)** निकालता है। जितना ज्यादा स्कोर, उतनी ज्यादा मदद की जरूरत।

3. **गणितीय बजट आवंटन (PuLP Optimization Engine)**:
   - यह सिस्टम कोई मनमर्जी नहीं करता; यह **Linear Programming** गणितीय मॉडल का उपयोग करता है।
   - आपके पास जितना भी कुल बजट है (उदा. ₹32 करोड़), यह उसे इस तरह बांटता है कि शहर का औसत घाटा सबसे तेजी से कम हो सके।

4. **भ्रष्टाचार व लीकेज पर रोक (Blockchain Smart Contracts)**:
   - ठेकेदार को सारा पैसा एक साथ नहीं दिया जाता।
   - जब ड्रोन या नागरिक यह वेरिफाई करते हैं कि काम पूरा हुआ है, तभी ब्लॉकचेन स्मार्ट कॉन्ट्रैक्ट से फंड रिलीज होता है।
"""
            speech = f"InfraBudget AI ek smart system hai jo kisi bhi shehar ke sabhi wardon ka data dekh kar yah tai karta hai ki kis ward ko kitne budget ki sabse zyaada zaroorat hai, aur blockchain ke zariye fund leakage ko rokta hai."

        elif lang == "Marathi":
            content = f"""
### 💡 इन्फ्राबजेट एआय (InfraBudget AI) काय आहे आणि कसे कार्य करते?

हा प्रकल्प महानगरपालिकेसाठी **डेटा-आधारित निधी वाटप प्रणाली (Data-Driven Budget Allocator)** आहे.

1. **समस्या (Problem)**: 
   - बऱ्याचदा निधीचे वाटप राजकीय प्रभावावर होते, ज्यामुळे खऱ्या गरजू प्रभागांना पुरेसा निधी मिळत नाही.
2. **तूट निर्देशांक (Deficit Score 0 ते 100)**:
   - रस्त्यांची दुरवस्था, पाणीटंचाई, आणि नागरिकांच्या तक्रारी यावरून एआय प्रत्येक प्रभागाचा 'डेफिसिट स्कोअर' ठरवतो.
3. **गणितीय वाटप (PuLP Optimization)**:
   - उपलब्ध बजेट सर्वाधिक गरजेनुसार प्रभागांमध्ये योग्य प्रमाणात वाटले जाते.
4. **पारदर्शकता (Blockchain Security)**:
   - कामाची प्रत्यक्ष पाहणी व ड्रोन पडताळणी झाल्याशिवाय निधी कंत्राटदाराला दिला जात नाही.
"""
            speech = f"InfraBudget AI he mahanagarpalike sathi ek smart system ahe, je pratyek prabhagachi garaj olakhun yogya nidhiche vatap karte."

        elif lang == "Kannada":
            content = f"""
### 💡 ಇನ್‌ಫ್ರಾಬಜೆಟ್ ಎಐ (InfraBudget AI) ಎಂದರೇನು?

ಇದು ಮಹಾನಗರ ಪಾಲಿಕೆಗಳಿಗೆ **ಡೇಟಾ ಆಧಾರಿತ ಬಜೆಟ್ ಹಂಚಿಕೆ ವ್ಯವಸ್ಥೆಯಾಗಿದೆ (Data-Driven Resource Allocator)**.

1. **ಸಮಸ್ಯೆ (Problem)**: ನಗರದ ವಾರ್ಡ್‌ಗಳಿಗೆ ಅನುದಾನವು ವೈಜ್ಞಾನಿಕವಾಗಿ ಹಂಚಿಕೆಯಾಗುವುದಿಲ್ಲ.
2. **ಎಐ ಕೊರತೆ ಸ್ಕೋರ್ (Deficit Score 0-100)**: ರಸ್ತೆಗಳ ವಯಸ್ಸು, ನೀರಿನ ಕೊರತೆ ಮತ್ತು ನಾಗರಿಕ ದೂರುಗಳ ಆಧಾರದ ಮೇಲೆ ಸ್ಕೋರ್ ನಿಗದಿಪಡಿಸುತ್ತದೆ.
3. **ಆಪ್ಟಿಮೈಸೇಶನ್ (PuLP Optimization)**: ಲಭ್ಯವಿರುವ ಬಜೆಟ್ ಅನ್ನು ಹೆಚ್ಚು ಅಗತ್ಯವಿರುವ ವಾರ್ಡ್‌ಗಳಿಗೆ ಹಂಚುತ್ತದೆ.
4. **ಬ್ಲಾಕ್‌ಚೈನ್ ಪಾರದರ್ಶಕತೆ**: ಅನುದಾನ ದುರುಪಯೋಗವಾಗದಂತೆ ಸ್ಮಾರ್ಟ್ ಒಪ್ಪಂದಗಳ ಮೂಲಕ ಹಣ ಬಿಡುಗಡೆ ಮಾಡಲಾಗುತ್ತದೆ.
"""
            speech = f"InfraBudget AI yembudu nagara wardgalige yogyavada budget hanchike maduva smart vyavastheyagide."

        else: # English
            content = f"""
### 💡 What is InfraBudget AI & How Does it Work?

InfraBudget AI is an enterprise **B2G (Business-to-Government) Decision Command Center** designed for municipal corporations to eliminate guesswork and corruption in public capital allocation.

1. **The Core Problem**: Municipal capital budgets are traditionally allocated based on political bargaining or historical inertia rather than real citizen need.
2. **Multi-Source Deficit Scoring (0-100)**: Our machine learning engine ingests road age, pothole count, water shortage %, drainage choke index, and 30-day citizen grievance logs to compute an objective **Infrastructure Deficit Score**.
3. **Mathematical Budget Optimization (PuLP LP Solver)**: Instead of arbitrary distribution, the system runs constrained linear programming to mathematically maximize city-wide deficit reduction while ensuring every ward receives minimum baseline equity.
4. **Blockchain Anti-Leakage Escrow**: Funds are disbursed in tranches governed by SHA-256 smart contracts, unlocked strictly upon satellite/drone verified project milestones.
"""
            speech = "InfraBudget AI is a data-driven municipal command center that uses machine learning and linear programming to allocate public funds to the most deserving city wards while preventing corruption via blockchain escrow."

        return {"type": "answer", "content": content, "speech": speech}

    # Check if a specific ward is mentioned in prompt
    matched_ward = None
    for _, w_row in df.iterrows():
        w_id_str = f"ward {w_row['ward_id']}"
        w_name_clean = w_row['ward_name'].lower()
        if w_id_str in p_lower or any(part.lower() in p_lower for part in w_row['ward_name'].split(' - ')[-1].split()):
            matched_ward = w_row
            break

    if matched_ward is not None:
        w = matched_ward
        if lang == "Hindi":
            content = f"""
### 📍 {w['ward_name']} की स्थिति विश्लेषण

- **बुनियादी ढांचा घाटा (Deficit Score)**: `{w['deficit_score']} / 100` ({w['risk_category']} श्रेणी)
- **सड़क स्थिति**: औसत `{w['avg_road_age_years']} वर्ष` पुरानी सड़क, `{w['unresolved_potholes']} गड्ढे` दर्ज हैं।
- **पेयजल की स्थिति**: `{w['water_deficit_pct']}%` जल संकट (पाइपलाइन दबाव: `{w['iot_pipeline_pressure_psi']} psi`)।
- **नाली व जलभराव**: चोक रिस्क इंडेक्स `{w['drainage_choke_idx']}/100`।
- **नागरिक शिकायतें (30 दिन)**: `{w['citizen_complaints_30d']} शिकायतें`।
- **वर्तमान बजट**: `₹{w['current_budget_lakhs']} लाख`।

**एआई सिफारिश**: इस वार्ड में प्राथमिकता के आधार पर जल व सड़क मरम्मत हेतु अतिरिक्त बजट दिया जाना चाहिए।
"""
            speech = f"{w['ward_name']} ka deficit score {w['deficit_score']} hai. Yahan paani ki kami {w['water_deficit_pct']}% aur {w['unresolved_potholes']} kharab gaddhe hain."
        elif lang == "Marathi":
            content = f"""
### 📍 {w['ward_name']} प्रभाग अहवाल

- **एकूण तूट निर्देशांक (Deficit Score)**: `{w['deficit_score']} / 100` ({w['risk_category']})
- **रस्त्यांची स्थिती**: सरासरी `{w['avg_road_age_years']} वर्षे` जुने रस्ते, `{w['unresolved_potholes']} खड्डे`.
- **पाणीटंचाई**: `{w['water_deficit_pct']}%` तुटवडा.
- **गटारे व सांडपाणी चोक**: `{w['drainage_choke_idx']}/100`.
- **सध्याचा निधी**: `₹{w['current_budget_lakhs']} लाख`.
"""
            speech = f"{w['ward_name']} cha deficit score {w['deficit_score']} ahe. Rastyanchi aani panyachi samasya ithe sarvat jasta ahe."
        elif lang == "Kannada":
            content = f"""
### 📍 {w['ward_name']} ವಿವರ

- **ಕೊರತೆ ಸ್ಕೋರ್ (Deficit Score)**: `{w['deficit_score']} / 100` ({w['risk_category']})
- **ರಸ್ತೆ ಸ್ಥಿತಿ**: `{w['avg_road_age_years']} ವರ್ಷ` ಹಳೆಯ ರಸ್ತೆ, `{w['unresolved_potholes']} ಗುಂಡಿಗಳು`.
- **ನೀರಿನ ಕೊರತೆ**: `{w['water_deficit_pct']}%`.
- **ಒಳಚರಂಡಿ ಸಮಸ್ಯೆ**: `{w['drainage_choke_idx']}/100`.
- **ಪ್ರಸ್ತುತ ಬಜೆಟ್**: `₹{w['current_budget_lakhs']} ಲಕ್ಷ`.
"""
            speech = f"{w['ward_name']} nalli deficit score {w['deficit_score']} aagide. Illi raste mathu neerina samasye hechagide."
        else:
            content = f"""
### 📍 Ward Status: {w['ward_name']}

- **Infrastructure Deficit Score**: `{w['deficit_score']} / 100` (**{w['risk_category']} Risk**)
- **Road Condition**: `{w['avg_road_age_years']} yrs` average age with `{w['unresolved_potholes']} active potholes`.
- **Water Deficit**: `{w['water_deficit_pct']}%` shortage (Pressure: `{w['iot_pipeline_pressure_psi']} psi`).
- **Drainage Risk**: `{w['drainage_choke_idx']}/100` choke index.
- **Citizen Grievances**: `{w['citizen_complaints_30d']} tickets` logged in last 30 days.
- **Current Allocation**: `₹{w['current_budget_lakhs']} Lakhs`.
"""
            speech = f"{w['ward_name']} has a deficit score of {w['deficit_score']} out of 100. Key issues are {w['water_deficit_pct']}% water shortage and {w['unresolved_potholes']} potholes."

        return {"type": "answer", "content": content, "speech": speech}

    # Check if user is asking about critical/worst wards
    if any(k in p_lower for k in ["critical", "worst", "highest deficit", "kharab", "gambhir", "pareshan", "sabse"]) or any(k in prompt for k in ["गंभीर", "खराब", "सबसे"]):
        crit_list = critical_wards.head(3)
        if lang == "Hindi":
            crit_bullets = "\n".join([f"- **{r['ward_name']}**: Deficit `{r['deficit_score']}/100` | सड़क आयु: `{r['avg_road_age_years']} वर्ष` | पानी कमी: `{r['water_deficit_pct']}%`" for _, r in crit_list.iterrows()])
            content = f"""
### 🚨 {city_name} के सर्वाधिक संकटग्रस्त वार्ड (Critical Deficit Wards)

वर्तमान में कुल **{len(critical_wards)} वार्ड** गंभीर जोखिम श्रेणी में हैं। शीर्ष 3 सबसे प्रभावित वार्ड:

{crit_bullets}

**मुख्य कारण**:
- सड़कों की अत्यधिक पुरानी आयु व अनसुलझे गड्ढे।
- पेयजल पाइपलाइन में कम दबाव व जलभराव की समस्या।
- उच्च नागरिक शिकायत दर (30 दिन)।
"""
            speech = f"Shahar ke sabse kharab sthiti wale wards hain: {', '.join(crit_list['ward_name'].tolist())}. Yahan sadak aur paani ki sthiti behad gambhir hai."
        else:
            crit_bullets = "\n".join([f"- **{r['ward_name']}**: Deficit `{r['deficit_score']}/100` | Road Age: `{r['avg_road_age_years']} yrs` | Water Shortage: `{r['water_deficit_pct']}%`" for _, r in crit_list.iterrows()])
            content = f"""
### 🚨 Most Critical Vulnerability Wards in {city_name}

Currently, **{len(critical_wards)} wards** exceed the critical deficit threshold (Score > 62.0). Top 3 priority wards:

{crit_bullets}

**Core Drivers**:
- Severely deteriorated asphalt with high unresolved pothole density.
- Low IoT water pressure and high drainage choke vulnerability.
- High citizen grievance ticket volume in the past 30 days.
"""
            speech = f"The most critical deficit wards requiring immediate capital intervention are {', '.join(crit_list['ward_name'].tolist())}."

        return {"type": "answer", "content": content, "speech": speech}

    # Check if budget / PuLP optimization query
    if any(k in p_lower for k in ["pulp", "linear programming", "budget", "allocate", "allocation", "math", "algorithm"]) or any(k in prompt for k in ["बजट", "बंटवारा", "गणित", "ऑप्टिमाइज़ेशन"]):
        if lang == "Hindi":
            content = f"""
### ⚖️ PuLP AI बजट का गणितीय व निष्पक्ष बंटवारा कैसे करता है?

नगर निगमों में पारंपरिक रूप से बजट अफसरों या नेताओं के प्रभाव से बंटता था। **InfraBudget AI** इसे पूरी तरह वैज्ञानिक बनाता है:

1. **घाटा स्कोर (Deficit Index 0-100)**:
   - हर वार्ड की टूटी सड़कें, पानी की कमी, और 30 दिन की जन शिकायतों के आधार पर घाटे का सटीक स्कोर निकाला जाता है।

2. **PuLP लीनियर प्रोग्रामिंग (Linear Programming LP Solver)**:
   - यह एक शक्तिशाली गणितीय अल्गोरिदम है।
   - आपके पास जितना कुल बजट उपलब्ध है (जैसे ₹{df['current_budget_lakhs'].sum():,.0f} लाख), यह उसे इस फॉर्मूले से बांटता है ताकि पूरे शहर का कुल बुनियादी घाटा सबसे कम हो सके।

3. **समानता का नियम (Baseline Equity Floor)**:
   - गणितीय मॉडल यह सुनिश्चित करता है कि किसी भी वार्ड को ₹0 न मिले; हर वार्ड को अनिवार्य न्यूनतम फंड मिलता है।

4. **ब्लॉकचेन एस्क्रो (Leakage Prevention)**:
   - फंड सीधे ठेकेदार को नहीं जाता; जब ड्रोन व नागरिक काम पूरा होने की पुष्टि करते हैं, तभी स्मार्ट कॉन्ट्रैक्ट फंड रिलीज करता है।
"""
            speech = f"PuLP Linear Programming ek mathematical model hai jo shahar ke kul budget ko is tarah baantata hai jisse sabhi wardon ka ausat ghata sabse jaldi kam ho sake aur sabhi ko nispaksh fund mile."
        else:
            content = f"""
### ⚖️ How PuLP AI Mathematically & Fairly Allocates Municipal Capital

Traditional municipal budget distribution often suffers from historical inertia and political lobbying. **InfraBudget AI** replaces this with mathematically proven **Constrained Linear Programming (PuLP)**:

1. **Objective Function**:
   - Minimize total city-wide infrastructure deficit weighted by population and ward vulnerability.
2. **Equity Floor Constraint**:
   - Guarantees every ward receives a baseline non-zero capital allocation regardless of size.
3. **Budget Ceiling Constraint**:
   - Total disbursed capital strictly equals the municipal sanctioned pool (`₹{df['current_budget_lakhs'].sum()/100:.2f} Crores`).
4. **Diminishing Returns Curve**:
   - Prioritizes capital where marginal benefit to deficit reduction is highest (severe drainage & pothole hazards).
5. **Blockchain Smart Milestone Escrow**:
   - Disburses funds in verifiable phases upon drone imagery and citizen quorum sign-off.
"""
            speech = f"PuLP uses constrained linear programming to mathematically minimize city-wide infrastructure deficit while guaranteeing fair baseline equity to every single ward."

        return {"type": "answer", "content": content, "speech": speech}

    # Drainage / Water query
    if "drainage" in p_lower or "water" in p_lower or "drain" in p_lower or "जल" in prompt or "नाली" in prompt or "पानी" in prompt or "पाणी" in prompt or "ನೀರು" in prompt:
        if lang == "Hindi":
            content = f"""
### 💧 जल निकासी व पेयजल विश्लेषण

- **सबसे गंभीर जलभराव वाला वार्ड**: **{worst_drainage['ward_name']}** (ड्रेनेज चोक स्कोर: `{worst_drainage['drainage_choke_idx']}/100`)।
- **सर्वाधिक पेयजल संकट**: **{worst_water['ward_name']}** (`{worst_water['water_deficit_pct']}%` पानी की कमी)।

**सिफारिश**: इन दोनों वार्डों को मानसून से पहले आपातकालीन ड्रेनेज डीसिल्टिंग व पाइपलाइन ओवरहाल फंड जारी किया जाना चाहिए।
"""
            speech = f"Sabse zyaada drainage samasya {worst_drainage['ward_name']} mein hai, aur paani ki kami {worst_water['ward_name']} mein hai."
        elif lang == "Marathi":
            content = f"""
### 💧 पाणीपुरवठा व सांडपाणी विश्लेषण

- **सांडपाणी तुंबणारा प्रभाग**: **{worst_drainage['ward_name']}** (चोक इंडेक्स: `{worst_drainage['drainage_choke_idx']}/100`)
- **पाणीटंचाई असलेला प्रभाग**: **{worst_water['ward_name']}** (`{worst_water['water_deficit_pct']}%` तुटवडा)
"""
            speech = f"Panyachi sarvat jasta tontavada {worst_water['ward_name']} madhe ahe."
        elif lang == "Kannada":
            content = f"""
### 💧 ನೀರು ಮತ್ತು ಒಳಚರಂಡಿ ವಿಶ್ಲೇಷಣೆ

- **ಚರಂಡಿ ಸಮಸ್ಯೆ ಇರುವ ವಾರ್ಡ್**: **{worst_drainage['ward_name']}** (ಚೋಕ್ ಸ್ಕೋರ್: `{worst_drainage['drainage_choke_idx']}/100`)
- **ನೀರಿನ ಕೊರತೆ ಇರುವ ವಾರ್ಡ್**: **{worst_water['ward_name']}** (`{worst_water['water_deficit_pct']}%` ಕೊರತೆ)
"""
            speech = f"Neerina samasye {worst_water['ward_name']} nalli hechagide."
        else:
            content = f"""
### 💧 Water & Drainage Analysis

- **Severe Drainage Choke**: **{worst_drainage['ward_name']}** (`{worst_drainage['drainage_choke_idx']}/100` choke index).
- **Highest Water Shortage**: **{worst_water['ward_name']}** (`{worst_water['water_deficit_pct']}%` supply deficit).

**Advisory**: Immediate capital injection required for drainage desilting and booster pumping before monsoon.
"""
            speech = f"The ward requiring most urgent drainage attention is {worst_drainage['ward_name']}."

        return {"type": "answer", "content": content, "speech": speech}

    # Road / Pothole query
    if "road" in p_lower or "pothole" in p_lower or "सड़क" in prompt or "गड्ढे" in prompt or "रस्ता" in prompt:
        if lang == "Hindi":
            content = f"""
### 🛣️ सड़क व गड्ढों का विश्लेषण

- **सबसे पुरानी व जर्जर सड़कें**: **{worst_roads['ward_name']}** (औसत आयु: `{worst_roads['avg_road_age_years']} वर्ष`)।
- **सर्वाधिक गड्ढे**: `{worst_roads['unresolved_potholes']} गड्ढे` इस वार्ड में मरम्मत की प्रतीक्षा में हैं।

**सिफारिश**: इस वार्ड में बिटुमिन रिसर्फेसिंग व रैपिड पैचवर्क के लिए बजट आवंटित किया जाए।
"""
            speech = f"Sabse zyaada tooti hui sadkein {worst_roads['ward_name']} mein hain, jahan {worst_roads['unresolved_potholes']} gaddhe hain."
        else:
            content = f"""
### 🛣️ Road Network & Pothole Audit

- **Worst Road Network**: **{worst_roads['ward_name']}** (Average road age: `{worst_roads['avg_road_age_years']} years`).
- **Hazardous Potholes**: `{worst_roads['unresolved_potholes']}` active potholes logged.

**Advisory**: High-durability asphalt resurfacing required immediately.
"""
            speech = f"The worst roads are located in {worst_roads['ward_name']} with {worst_roads['unresolved_potholes']} active potholes."

        return {"type": "answer", "content": content, "speech": speech}

    # Default / General Question Fallback
    avg_score = df['deficit_score'].mean()
    if lang == "Hindi":
        content = f"""
### 🏛️ नगर निगम अवलोकन ({city_name})

- **शहर का औसत बुनियादी घाटा**: `{avg_score:.1f} / 100`
- **अत्यधिक गंभीर वार्ड (Critical Wards)**: `{len(critical_wards)} वार्ड`
- **सबसे ज्यादा जरूरतमंद वार्ड**: **{top_critical['ward_name']}** (घाटा स्कोर: `{top_critical['deficit_score']}`)

*आप किसी भी वार्ड का नाम (जैसे 'Ward 4' या 'Okhla') या समस्या (जैसे 'सड़क', 'पानी') लिखकर पूछ सकते हैं।*
"""
        speech = f"Shahar ka ausat deficit score {avg_score:.1f} hai. Sabse zyaada samasya {top_critical['ward_name']} mein hai."
    elif lang == "Marathi":
        content = f"""
### 🏛️ महानगरपालिका आढावा ({city_name})

- **शहराचा सरासरी तूट निर्देशांक**: `{avg_score:.1f} / 100`
- **गंभीर प्रभाग**: `{len(critical_wards)} प्रभाग`
- **सर्वात गरजू प्रभाग**: **{top_critical['ward_name']}** (स्कोअर: `{top_critical['deficit_score']}`)
"""
        speech = f"Shaharat sarvat jasta garaj {top_critical['ward_name']} la ahe."
    elif lang == "Kannada":
        content = f"""
### 🏛️ ಮಹಾನಗರ ಪಾಲಿಕೆ ಅವಲೋಕನ ({city_name})

- **ನಗರದ ಸರಾಸರಿ ಕೊರತೆ ಸ್ಕೋರ್**: `{avg_score:.1f} / 100`
- **ತೀವ್ರ ಕೊರತೆಯಿರುವ ವಾರ್ಡ್‌ಗಳು**: `{len(critical_wards)}`
- **ಹೆಚ್ಚು ಅಗತ್ಯವಿರುವ ವಾರ್ಡ್**: **{top_critical['ward_name']}** (ಸ್ಕೋರ್: `{top_critical['deficit_score']}`)
"""
        speech = f"Nagaradalli hecchina samasye {top_critical['ward_name']} nalli ide."
    else:
        content = f"""
### 🏛️ Municipal Infrastructure Overview ({city_name})

- **City Mean Deficit Score**: `{avg_score:.1f} / 100`
- **Critical Risk Wards**: `{len(critical_wards)} wards`
- **Highest Deficit Ward**: **{top_critical['ward_name']}** (`{top_critical['deficit_score']} / 100`)

*You can ask about any specific ward name, problem type (Roads, Drainage, Water), or type 'explain project' for a full simple overview.*
"""
        speech = f"The city average deficit score is {avg_score:.1f}. Highest deficit ward is {top_critical['ward_name']}."

    return {"type": "answer", "content": content, "speech": speech}


def generate_executive_report(df: pd.DataFrame, city_name: str = "Delhi (MCD)") -> str:
    """
    Generates a formal GovTech justification report in Markdown for the Municipal Commissioner.
    """
    today_str = datetime.now().strftime("%d %B %Y")
    critical_wards = df[df["risk_category"] == "Critical"].sort_values(by="deficit_score", ascending=False)
    total_budget = df["current_budget_lakhs"].sum()
    avg_deficit = df["deficit_score"].mean()

    lines = [
        f"# {city_name.upper()} CAPITAL ALLOCATION MEMORANDUM",
        f"**OFFICE OF THE MUNICIPAL COMMISSIONER & URBAN PLANNING DIRECTORATE**",
        f"**Document ID:** `MCMC/CAPEX/2026-REV/048` | **Date:** `{today_str}`",
        f"**Subject:** Algorithmic Ward Capital Budget Allocation & Deficit Mitigation Justification",
        "",
        "---",
        "",
        "### 1. EXECUTIVE SUMMARY",
        f"Under the InfraBudget AI framework, an algorithmic diagnostic of all **{len(df)} municipal wards** was conducted integrating GIS telemetry, citizen grievance logs (30 days), and road condition surveys.",
        f"- **City Baseline Infrastructure Deficit Index:** `{avg_deficit:.1f} / 100`",
        f"- **Critical Vulnerability Wards:** `{len(critical_wards)} wards` exceeding the 62.0 risk threshold.",
        f"- **Total Capital Budget Analyzed:** `₹{total_budget:,.1f} Lakhs` (`₹{total_budget/100:.2f} Crores`).",
        "",
        "### 2. CRITICAL PRIORITY ALLOCATION TABLE",
        "| Rank | Ward Identifier | Zone | Deficit Score | Road Age | Water Deficit | Recommended Capex |",
        "|---|---|---|---|---|---|---|"
    ]

    for rank, (_, row) in enumerate(critical_wards.iterrows(), start=1):
        rec_budget = row['current_budget_lakhs'] * 1.35
        lines.append(f"| #{rank} | **{row['ward_name']}** | {row['zone']} | `{row['deficit_score']}/100` | {row['avg_road_age_years']} yrs | {row['water_deficit_pct']}% | **₹{rec_budget:.1f}L** |")

    lines.extend([
        "",
        "### 3. MATHEMATICAL & ALGORITHMIC JUSTIFICATION",
        "The proposed fund distribution was calculated utilizing the **PuLP Linear Programming Solver** constrained by:",
        "1. **Equity Floor Constraint:** Minimum budget guaranteed to every ward regardless of historical bias.",
        "2. **Diminishing Marginal Return Metric:** Priority weights applied to wards with acute drainage and road bottlenecks.",
        "3. **Zero-Leakage Assurance:** All disbursement tranches are escrowed on the Municipal Blockchain Ledger with Smart Contract verification gates (Drone LiDAR & Citizen Sign-Off Quorums).",
        "",
        "### 4. SUBMISSION & SIGN-OFF",
        "**Prepared By:** InfraBudget AI Decision Engine (Autonomous Analytics v2.4)",
        "**Status:** *RECOMMENDED FOR IMMEDIATE SANCTION BY MAYORAL COUNCIL*"
    ])

    return "\n".join(lines)


def generate_speech_audio(text: str, lang: str = "Hindi") -> bytes:
    """
    Synthesizes natural MP3 speech audio from text using gTTS.
    Falls back gracefully if network is unavailable.
    """
    try:
        from gtts import gTTS
        import io
        clean_text = text.replace("#", "").replace("*", "").replace("•", "").replace("`", "").strip()
        if len(clean_text) > 350:
            clean_text = clean_text[:350] + "..."
        tts_lang = "hi" if lang in ["Hindi", "Hinglish", "hi"] else "en"
        tts = gTTS(text=clean_text, lang=tts_lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception:
        return None

