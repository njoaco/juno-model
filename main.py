import os
import subprocess
from utils.logo import print_logo

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main_menu():
    clear_screen()
    print_logo()

    print("Select asset type:")
    print("1. Cryptocurrencies")
    print("2. Stocks")
    print("3. Exit\n")
    
    while True:
        choice = input("Select an option (1-3): ")
        if choice in ['1', '2', '3']:
            return choice
        print("Invalid option. Please try again.")

def action_menu(asset_type):
    clear_screen()
    print_logo()

    print(f"Select an action for {asset_type}:")
    print("1. Train new model")
    print("2. Make prediction")
    print("3. Back\n")
    
    while True:
        choice = input("Select an option (1-3): ")
        if choice in ['1', '2', '3']:
            return choice
        print("Invalid option. Please try again.")

def main():
    while True:
        asset_choice = main_menu()
        
        if asset_choice == '3':
            print("\nGoodbye!")
            break

        asset_type = "Cryptocurrencies" if asset_choice == '1' else "Stocks"
        
        while True:
            action_choice = action_menu(asset_type)
            
            if action_choice == '3':
                break
            
            script = 'scripts/train_model.py' if action_choice == '1' else 'scripts/prediction.py'
            subprocess.run(['python', script, asset_choice])
            
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
