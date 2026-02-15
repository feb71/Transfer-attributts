# Transfer-attributts
Her er en **kort og praktisk brukerveiledning** du kan gi til sluttbruker 👇

---

# VA Linje-Matcher

**Overføring av teoretiske attributter til innmålte linjer**

---

## 🎯 Hva gjør programmet?

Programmet sammenligner:

* **Teoretiske VA-linjer**
* **Innmålte VA-linjer**

og kopierer riktige attributtverdier fra teoretisk datasett til innmålt datasett.

Geometrien i innmålt datasett endres ikke.

Resultatet er en ny Excel-fil som kan importeres tilbake til Gemini.

---

## 📥 Hva trenger du?

To Excel-filer:

1. **Teoretisk datasett**
2. **Innmålt datasett**

Begge må inneholde:

* `Id` (linje-ID)
* `Nr.` (punktrekkefølge)
* `Øst`
* `Nord`

Gemini-eksport støttes direkte.

---

## ⚙️ Hvordan bruke programmet

### 1️⃣ Last opp filer

* Last opp teoretisk Excel
* Last opp innmålt Excel

---

### 2️⃣ Velg match-innstillinger

**Bufferavstand (meter)**
Hvor nær linjene må ligge (f.eks. 1.0 m)

**Antall snittpunkter**
Hvor mange punkter langs linjen som skal kontrolleres

**Minimum treff**
Hvor mange av snittpunktene som må ligge innenfor buffer

---

### 3️⃣ Velg match-attributter

Velg hvilke felt som må stemme, f.eks:

* Type ledning (VL / SP / OV)
* Dimensjon

---

### 4️⃣ Velg hvilke attributter som skal overføres

Du kan:

* Overføre alle
* Eller velge ut bestemte felt

---

### 5️⃣ Kjør matching

Trykk **“Kjør matching”**

---

## 📤 Resultat

Du får en Excel-fil med:

* Samme struktur som innmålt datasett
* Utfylte attributter
* Egen fane med match-rapport

ID-strukturen beholdes i Gemini-format
(ID kun på første punkt per linje)

---

## 📊 Match-rapport

Rapporten viser per linje:

* Matchet / Ikke matchet
* Hvilken teoretisk linje som ble valgt
* Antall treff langs linja

Linjer uten treff må vurderes manuelt.

---

## ⚠️ Viktig

* Begge datasett må være i samme koordinatsystem
* Bufferen bør normalt være 0.5–1.0 meter
* Hvis mange feilmatchinger: reduser buffer eller øk krav til treff

---

## 🚀 Typisk arbeidsflyt

1. Eksporter fra Gemini
2. Kjør match i programmet
3. Last ned resultat
4. Importer tilbake i Gemini


