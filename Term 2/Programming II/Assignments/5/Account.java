package hw5;

import java.util.ArrayList;
import java.util.Date;

public class Account {
	
	private String name;
	private int accountNumber;
	private double balance;
	private double annualInterest;
	private Date dateCreated;
	private ArrayList<Transaction> transactions;
	
	public Account() {
		dateCreated = new Date();
		transactions = new ArrayList<Transaction>();
	}
	
	public Account(String name, int accountNumber, double balance) {
		this.name = name;
		this.accountNumber = accountNumber;
		this.balance = balance;
		dateCreated = new Date();
		transactions = new ArrayList<Transaction>();
	}
	
	public void deposit(double amount) {
		balance = balance + amount;
		transactions.add(new Transaction(new Date(), 'D', amount, balance));
		System.out.println(amount + " deposited successfully");
	}
	
	public void withdraw(double amount) {
		balance = balance - amount;
		transactions.add(new Transaction(new Date(), 'W', amount, balance));
		System.out.println(amount + " withdrawn successfully");
	}
	
	public String toString() {
		return ("Account owner: " + name + "\nAccount number: " + accountNumber + 
				"\nCurrent balance: " + balance + "\nAnnual interest: " + annualInterest + "%");
	}
	
	public void displayTransactions() {
		System.out.println(transactions.toString());
	}
	
	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public int getAccountNumber() {
		return accountNumber;
	}

	public void setAccountNumber(int accountNumber) {
		this.accountNumber = accountNumber;
	}

	public double getBalance() {
		return balance;
	}

	public void setBalance(double balance) {
		this.balance = balance;
	}
	
	public double getAnnualInterest() {
		return annualInterest;
	}

	public void setAnnualInterest(double annualInterest) {
		this.annualInterest = annualInterest;
	}

	public Date getDateCreated() {
		return dateCreated;
	}

	public void setDateCreated(Date dateCreated) {
		this.dateCreated = dateCreated;
	}
	
}
