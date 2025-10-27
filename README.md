# Universal-and-Transferable-Attacks-on-Pathology-Foundation-Models

*Unofficial exploration of Universal and Transferable Attacks on Pathology Foundation Models.*

---

### Perturbation trained for **UNI**

<img width="224" height="224" alt="utap perturbation visualization (UNI)" src="https://github.com/user-attachments/assets/75ec838d-897c-4e61-b31f-c8c2b24ec39b" />

> **Note:** Perturbation trained for **UNI**.

---

### Before and After addition of Perturbation **[UNI]**

> I tried the original parameters but found that the addition of the perturbation is quite visible (likely a flaw in my training setup).

<img width="1490" height="794" alt="STR-TCGA-LLMGHAHA pair (UNI)" src="https://github.com/user-attachments/assets/f521c099-5b50-4ea8-95f9-3045b6412e73" />
<img width="1490" height="794" alt="STR-TCGA-VEMARASN pair (UNI)" src="https://github.com/user-attachments/assets/c35240a4-6e19-41d0-a9e2-e583acf77992" />
<img width="1490" height="794" alt="TUM-TCGA-YPPSLHGS pair (UNI)" src="https://github.com/user-attachments/assets/c315effc-a329-43c6-9022-3d514c9411a2" />
<img width="1490" height="794" alt="LYM-TCGA-ENRAPMQS pair (UNI)" src="https://github.com/user-attachments/assets/abae3c86-7164-4bff-a5f8-7d90b8d6c8d6" />

---

### Accuracy on **CRC-VAL-HE-7K** using **UNI**

<img width="1600" height="900" alt="Accuracy (clean, 1 epoch) – UNI" src="https://github.com/user-attachments/assets/34db3e28-dedf-4886-8886-695d0c017dfe" />

---

### Internal attacks (Perturbation trained for the **same model**)

**Accuracy on CRC-VAL-HE-7K using UNI *after* addition of Perturbation [UNI]:**

<img width="1600" height="900" alt="Accuracy (attacked, 1 epoch) – UNI" src="https://github.com/user-attachments/assets/ab5ccb50-a8d8-4704-bf17-6b5f49b7c0f7" />

> We see a **huge drop** in accuracies after addition of the perturbation.

**Accuracy on CRC-VAL-HE-7K using UNI after addition of Perturbation [UNI], with a simple FFT-based pre-processing that attenuates the attack:** We convert each 224×224 image to the frequency domain via a 2D FFT, where meaningful content concentrates near the center (low frequencies) and adversarial noise tends to occupy the periphery (high frequencies). We then apply a circular low-pass mask (e.g., cutoff radius ≈ 50 px) to preserve the central band while suppressing the outer high-frequency components that carry most of the perturbation energy. After masking, we run the inverse FFT, take the real component, clamp to a valid range, normalize, and feed the result to a frozen UNI2-h (ViT-G) feature extractor with a linear probe. This simple pre-processing reduces the attack’s effect by filtering out high-frequency artifacts while retaining class-relevant structure, enabling a measurable recovery in accuracy.
**

<img width="1600" height="900" alt="Accuracy (defended via FFT) – UNI" src="https://github.com/user-attachments/assets/5360d9f9-db32-448e-9389-1ed7b7f75478" />

---

### UMAP and PCA — **UNI** (before and after Perturbation [UNI])

<img width="2823" height="1255" alt="UNI UMAP 2D (before vs after)" src="https://github.com/user-attachments/assets/4ed9001d-a7ec-4716-ba0c-7c0e72c89fc1" />
<img width="2823" height="1255" alt="UNI PCA 2D (before vs after)" src="https://github.com/user-attachments/assets/12dc239d-5e23-4c70-a30c-0ea2161b73a2" />

---

### Perturbation trained for **CONCH**

<img width="224" height="224" alt="utap perturbation visualization (CONCH)" src="https://github.com/user-attachments/assets/1cb165a9-0c7d-4ddb-8f78-d01f4934bb95" />

> **Note:** Perturbation trained for **CONCH**.

---

### Accuracy on **CRC-VAL-HE-7K** using **CONCH** (simple prompt template)

<img width="1800" height="1000" alt="Accuracy (clean) – CONCH" src="https://github.com/user-attachments/assets/599ce240-91f5-499e-8480-9133b4b26a93" />

---

### Before and After addition of Perturbation **[CONCH]**

<img width="1490" height="794" alt="LYM-TCGA-ENRAPMQS pair (CONCH)" src="https://github.com/user-attachments/assets/c4af0e80-dfa3-4f3c-9aea-de524791f3c1" />
<img width="1490" height="794" alt="TUM-TCGA-YPPSLHGS pair (CONCH)" src="https://github.com/user-attachments/assets/46257b6c-126f-44d8-84cb-88d74b324474" />
<img width="1490" height="794" alt="STR-TCGA-VEMARASN pair (CONCH)" src="https://github.com/user-attachments/assets/19f6d90e-e80a-4120-a755-613ce80d91cb" />
<img width="1490" height="794" alt="STR-TCGA-LLMGHAHA pair (CONCH)" src="https://github.com/user-attachments/assets/305cfcf1-eee6-4569-b41a-03d1334d6d42" />

---

### UMAP and PCA — **CONCH** (before and after Perturbation [CONCH])

<img width="2823" height="1255" alt="CONCH PCA 2D (before vs after)" src="https://github.com/user-attachments/assets/f9e4a5c9-9226-48a4-94d0-96ecdf77fe6d" />
<img width="2823" height="1255" alt="CONCH UMAP 2D (before vs after)" src="https://github.com/user-attachments/assets/7ca17c5f-080a-4819-8ab2-60c4c0265d23" />

---

### Accuracy on **CRC-VAL-HE-7K** using **CONCH** *after* addition of Perturbation **[CONCH]**

<img width="1800" height="1000" alt="Accuracy (attacked) – CONCH" src="https://github.com/user-attachments/assets/f8ef815e-b484-466f-9af4-6eb39c7a0506" />

---

### External attacks (Perturbation trained for a **different model**)

**Accuracy on CRC-VAL-HE-7K using CONCH after addition of Perturbation [UNI]:**

<img width="1800" height="1000" alt="Accuracy – CONCH under UNI perturbation" src="https://github.com/user-attachments/assets/77434682-fdfc-4d62-9052-96b4e051bb39" />

> Not a **significant drop** in VLMs (some classes decrease while others increase), under external attack.

**UMAP and PCA — CONCH (before and after Perturbation [UNI])**

<img width="2823" height="1255" alt="CONCH UMAP 2D (UNI perturbation)" src="https://github.com/user-attachments/assets/3db17370-7681-4a38-84a4-bb457f32b4ca" />
<img width="2823" height="1255" alt="CONCH PCA 2D (UNI perturbation)" src="https://github.com/user-attachments/assets/87a92b47-ece9-404b-8a55-73e230868f10" />

**Accuracy on CRC-VAL-HE-7K using UNI after addition of Perturbation [CONCH]:**

<img width="2400" height="1350" alt="Accuracy – UNI under CONCH perturbation" src="https://github.com/user-attachments/assets/51875686-b5a4-4ea9-805a-c578853ce20f" />

> We see a drop, but **not a significant** one.

<!-- (dont change the ordering of the information...) -->
