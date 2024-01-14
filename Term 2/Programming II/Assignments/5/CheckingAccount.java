package hw5;

public class CheckingAccount extends Account {
	
	private final static double OVERDRAFT_LIMIT = 1000;
	private double overdraftAmount;
	
	public CheckingAccount() {
		super();
	}
	
	public CheckingAccount(String name, int accountNumber, double balance) {
		super(name, accountNumber, balance);
	}
	
	public void withdraw(double amount) {
		double balance = getBalance();
		if(balance < amount) {
			if(overdraftAmount + amount - balance > OVERDRAFT_LIMIT) {
				System.out.println("Overdraft limit exeeded. Could not withdraw required amount.");
			}
			else {
				
				super.withdraw(balance);
				overdraftAmount = overdraftAmount + (amount - balance);
				System.out.println((amount - balance) + " overdrawn");
				System.out.println("Overdraft Amount: " + overdraftAmount);
			}
		}
		else {
			super.withdraw(amount);
		}
		
	}
	
	public void deposit(double amount) {
		if(overdraftAmount >= amount) {
			overdraftAmount = overdraftAmount - amount;
		}
		else {
			amount = amount - overdraftAmount;
			overdraftAmount = 0;
			super.deposit(amount);
		}
		
	}
	
	public String toString() {
		return ("Account Type: Checking Account\n" + super.toString() + "\nOverdraft amount: " + overdraftAmount);
	}
	

}
