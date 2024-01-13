package project;

import java.io.IOException;

public class SavingsAccount extends Account {
	
	
	public SavingsAccount(String name, int accountNumber, double balance) throws IOException {
		super(name, accountNumber, balance);
	}
	
	public void withdraw(double amount) {
		double balance = getBalance();
		if(balance < amount) {
			System.out.println("Cannot withdraw required ammount. Insufficient balance in account.");
		}
		else {
			super.withdraw(amount);
		}
	}
	
	public String toString() {
		return ("Account Type: Savings Account\n" + super.toString());
	}
	

}
