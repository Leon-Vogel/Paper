import pandas as pd

Reward = {
    'Erfüllung des Produktionsplan': {'Strafe für X wenn Soll X erreicht ist und Y...Z noch nicht': 'V1',
                                      'Belohnung für alle / den Anteil nach Plan gefertigter Produkte': 'V2, V20'},
    'Durchlaufzeit/Makespan': {'Belohnung für niedrige Durchlaufzeit, Bsp. + 0.2*(2-MittDlz/Ref)': ''},
    'Auslastung': {'Belohnung für Mittelwert oder Varianz der Auslastung': '',
                   'Strafe für Varianz der Varianzen': 'V4',
                   'Strafe für Varianz von fertigen Produkten seit letztem Step': '',
                   'evtl. Summe der Varianz je Station?': ''},
    'Warteschlangen': {'Strafe für Anzahl auf Förderstrecke >2': 'V0, V20',
                       'Strafe für Anzahl auf Förderstrecke >1': 'V00',
                       'Strafe für Varianz / Mittelwert der Anzahl P auf Förderstrecke': 'V5'},
    'Leerlaufzeit': {'Strafe für Varianz oder Mittelwert der Steps seit letztem P': '',
                     'Strafe für Varianz der Varianzen': 'V3'}
}

for key, values in Reward:
    print(str(key)+':\n'+str(values)+'\n\n')
