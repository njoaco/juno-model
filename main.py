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

def model_menu(action, asset_type):
    clear_screen()
    print_logo()
    print(f"Select model type for {action} {asset_type}:")
    print("1. Normal Model")
    print("2. Mini Model\n")
    
    while True:
        choice = input("Select an option (1-2): ")
        if choice in ['1', '2']:
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
            
            action_text = "training" if action_choice == '1' else "prediction"
            model_choice = model_menu(action_text, asset_type)
            
            if action_choice == '1':  # Training
                script = 'scripts/train_model.py' if model_choice == '1' else 'scripts/train_model_mini.py'
            else:  # Prediction
                script = 'scripts/prediction.py' if model_choice == '1' else 'scripts/prediction_mini.py'
            
            subprocess.run(['python', script, asset_choice])
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
