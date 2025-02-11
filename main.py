import os
import subprocess
from utils.logo import print_logo

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main_menu():
    clear_screen()
    print_logo()

    print("Seleccione el tipo de activo:")
    print("1. Criptomonedas")
    print("2. Acciones (Stocks)")
    print("3. Salir\n")
    
    while True:
        choice = input("Seleccione una opción (1-3): ")
        if choice in ['1', '2', '3']:
            return choice
        print("Opción inválida. Intente nuevamente.")

def action_menu(asset_type):
    clear_screen()
    print_logo()

    print(f"Seleccione una acción para {asset_type}:")
    print("1. Entrenar nuevo modelo")
    print("2. Realizar predicción")
    print("3. Volver\n")
    
    while True:
        choice = input("Seleccione una opción (1-3): ")
        if choice in ['1', '2', '3']:
            return choice
        print("Opción inválida. Intente nuevamente.")

def main():
    while True:
        asset_choice = main_menu()
        
        if asset_choice == '3':
            print("\n¡Hasta luego!")
            break
        
        asset_type = "Criptomonedas" if asset_choice == '1' else "Acciones (Stocks)"
        
        while True:
            action_choice = action_menu(asset_type)
            
            if action_choice == '3':
                break
            
            script = 'scripts/train_model.py' if action_choice == '1' else 'scripts/prediction.py'
            subprocess.run(['python', script, asset_choice])
            
            input("\nPresione Enter para continuar...")

if __name__ == "__main__":
    main()
