import torch
from flask import Flask, request, render_template
from transformers import AutoModelForSequenceClassification, DistilBertTokenizer

app = Flask(__name__)

path = "results/checkpoint-25000"
model = AutoModelForSequenceClassification.from_pretrained(path)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

classes = ['T1005', 'T1010', 'T1027', 'T1059', 'T1074', 'T1105', 'T1112', 'T1564.001', 
           'T1204.002', 'T1071.001', 'T1041', 'T1071', 'T1132', 'T1056.001', 'T1056', 
           'T1203', 'T1053.005', 'T1053', 'T1057', 'T1026', 'T1113', 'T1070.004', 
           'T1553.002', 'T1574.002', 'T1047', 'T1090.003', 'T1055.013', 'T1574.001', 
           'T1114', 'T1021.002', 'T1012', 'T1547.001', 'T1547.009', 'T1546.008', 
           'T1030', 'T1003', 'T1078', 'T1133', 'T1049', 'T1007', 'T1083', 'T1087', 
           'T1036', 'T1016', 'T1046', 'T1001', 'T1098', 'T1543.001', 'T1037.004', 
           'T1543.003', 'T1124', 'T1064', 'T1082', 'T1033', 'T1025', 'T1486', 
           'T1218.011', 'T1055', 'T1070', 'T1104', 'T1102', 'T1095', 'T1571', 
           'T1518.001', 'T1120', 'T1090', 'T1140', 'T1037', 'T1548.002', 'T1055.012', 
           'T1027.002', 'T1027.001', 'T1027.005', 'T1043', 'T1204', 'T1598.003', 
           'T1021.001', 'T1068', 'T1119', 'T1020', 'T1197', 'T1573', 'T1127', 
           'T1008', 'T1562.001', 'T1546.007', 'T1048', 'T1569.002', 'T1018', 
           'T1553.004', 'T1202', 'T1560', 'T1136', 'T1021.003', 'T1547.004', 
           'T1546.002', 'T1546.009', 'T1059.001', 'T1135', 'T1129', 'T1070.006', 
           'T1134', 'T1552.001', 'T1069', 'T1550.002', 'T1123', 'T1125', 'T1115', 
           'T1029', 'T1557.001', 'T1559.002', 'T1210', 'T1218.005', 'T1222', 
           'T1106', 'T1547.002', 'T1021.006', 'T1040', 'T1555.001', 'T1201', 
           'T1505.003', 'T1189', 'T1072', 'T1110', 'T1034', 'T1091', 'T1080', 
           'T1052', 'T1218.010', 'T1092', 'T1547.008', 'T1217', 'T1598.002', 
           'T1108', 'T1190', 'T1213', 'T1550.003', 'T1542.003', 'T1070.005', 
           'T1039', 'T1187', 'T1221', 'T1061', 'T1552.002', 'T1546.011', 
           'T1218.003', 'T1546.015', 'T1211', 'T1199', 'T1021', 'T1090.004', 
           'T1137', 'T1546.003', 'T1014', 'T1564.004', 'T1558.003', 'T1542.001', 
           'T1056.004', 'T1111', 'T1542.002', 'T1218.001', 'T1547.015', 
           'T1205.001', 'T1036.006', 'T1546.010', 'T1566.003', 'T1219', 'T1220', 
           'T1059.002', 'T1555.002', 'T1185', 'T1195', 'T1207', 'T1543', 
           'T1546.013', 'T1497.001', 'T1497', 'T1490', 'T1482', 'T1561.002', 
           'T1548.001', 'T1552.004', 'T1218.002', 'T1556.002', 'T1011', 
           'T1134.005', 'T1548.003', 'T1574.010', 'T1547.005', 'T1485', 
           'T1216', 'T1055.011', 'T1569.001']

def predict_multilabel(text, threshold=0):
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        logits_np = logits.cpu().numpy()
        predictions = (logits_np > threshold).astype(int)

    predicted_indices = predictions[0].nonzero()[0].tolist()
    predicted_classes = [classes[idx] for idx in predicted_indices]

    return predicted_classes

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form.get('text')
        if text:
            pred = predict_multilabel(text)
            return render_template("main.html", prediction=pred)
    return render_template("main.html")

if __name__ == "__main__":
    app.run(debug=True)

