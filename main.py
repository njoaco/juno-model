# main.py
import os
import subprocess
from utils.logo import print_logo

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main_menu():
    clear_screen()
    print_logo()

    print("1. Entrenar nuevo modelo")
    print("2. Realizar predicción")
    print("3. Salir\n")
    
    while True:
        choice = input("Seleccione una opción (1-3): ")
        if choice in ['1', '2', '3']:
            return choice
        print("Opción inválida. Intente nuevamente.")

def main():
    while True:
        option = main_menu()
        
        if option == '1':
            subprocess.run(['python', 'scripts/train_model.py'])
        elif option == '2':
            subprocess.run(['python', 'scripts/prediction.py'])
        elif option == '3':
            print("\n¡Hasta luego!")
            break

        input("\nPresione Enter para continuar...")

if __name__ == "__main__":
    main()