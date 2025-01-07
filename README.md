Solutie BERT

1. Încărcarea și împărțirea datelor
Datele din fișierul train.csv au fost împărțite în seturi de antrenare și validare (80% antrenare, 20% validare). Fișierul test.csv a fost încărcat pentru a face predicții asupra articolelor necunoscute.

2. Maparea etichetelor
Etichetele textuale (fake, biased, true) au fost mapate la valori numerice:

fake → 0
biased → 1
true → 2
Această mapare este necesară pentru compatibilitatea cu modelul de învățare automată.

3. Tokenizarea textelor
Am utilizat tokenizer-ul de la CamemBERT pentru a converti textele în reprezentări numerice potrivite modelului. Tokenizarea s-a realizat cu:

truncation=True: pentru a limita textele la o lungime maximă de 512 tokeni.
padding=True: pentru alinierea dimensiunilor secvențelor.
4. Crearea dataset-urilor compatibile cu PyTorch
Encodările și etichetele au fost transformate în dataset-uri compatibile cu PyTorch pentru antrenare și validare.

5. Configurarea și antrenarea modelului
Modelul CamemBERT pre-antrenat a fost utilizat pentru clasificare. Am configurat următoarele parametri de antrenare:

Batch size: 8
Număr de epoci: 3
Strategia de salvare a modelului: la sfârșitul fiecărei epoci.
Funcția de evaluare a calculat acuratețea, precizia, recall-ul și scorul F1 pentru a monitoriza performanța.

6. Predicția etichetelor pentru setul de testare
Modelul antrenat a fost utilizat pentru a prezice etichetele din fișierul test.csv. Etichetele numerice au fost mapate înapoi la valori textuale (fake, biased, true).

7. Salvarea rezultatelor
Fișierul completat (test_completed_bert.csv) conține textele din setul de testare și etichetele prezise.

8. Salvarea modelului și tokenizer-ului
Modelul și tokenizer-ul au fost salvați pentru reutilizare ulterioară.

Accuratete: 0.82
