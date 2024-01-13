package project;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Date;

public class CheckingAccount extends Account {
	
	private final static double OVERDRAFT_LIMIT = 1000;
	private double overdraftAmount;
	
	
	public CheckingAccount(String name, int accountNumber, double balance) throws IOException {
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
				Transaction t = new Transaction(new Date(), "Overdraft Withdrawal", (amount-balance), getBalance(), "Total overdraft amount : " + overdraftAmount);
				writeTransaction(t);
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
			Transaction t = new Transaction(new Date(), "Overdraft Deposit", amount, getBalance(), "Total overdraft amount : " + overdraftAmount);
			writeTransaction(t);
		}
		else {
			amount = amount - overdraftAmount;
			Transaction t = new Transaction(new Date(), "Overdraft Deposit", overdraftAmount, getBalance(), "Total overdraft amount : 0");
			overdraftAmount = 0;
			super.deposit(amount);
		}
		
	}
	
	public String toString() {
		return ("Account Type: Checking Account\n" + super.toString() + "\nOverdraft amount: " + overdraftAmount);
	}
	

}
