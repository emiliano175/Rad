from fpdf import FPDF

def generate_pdf(name, age, gender, treatment_site, comorbidities, predictions):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "RadiRecover Patient Report", ln=True, align='C')

    pdf.ln(10)
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, f"Name: {name}", ln=True)
    pdf.cell(200, 10, f"Age: {age}", ln=True)
    pdf.cell(200, 10, f"Gender: {gender}", ln=True)
    pdf.cell(200, 10, f"Treatment Site: {treatment_site}", ln=True)
    pdf.cell(200, 10, f"Comorbidities: {', '.join(comorbidities)}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "Predicted Side Effects:", ln=True)
    pdf.set_font("Arial", '', 12)
    for label, prob in predictions.items():
        pdf.cell(200, 10, f"{label}: {prob:.1f}%", ln=True)

    filepath = "patient_report.pdf"
    pdf.output(filepath)
    return filepath
