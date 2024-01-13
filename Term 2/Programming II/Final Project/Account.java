package project;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Date;
import java.util.Scanner;

public class Account {
	
	private String name;
	private int accountNumber;
	private double balance;
	private double annualInterest;
	private Date dateCreated;
	private File transactions;
	private PrintWriter writer;
	
	public Account(String name, int accountNumber, double balance) throws IOException {
		this.name = name;
		this.accountNumber = accountNumber;
		this.balance = balance;
		dateCreated = new Date();
		transactions = new File("C:/Users/M/Desktop/Project Files/" + accountNumber + "_transactions.txt");
		if(transactions.exists()) {
			transactions.delete();
		}
		writer = new PrintWriter(new FileWriter(transactions, true));
		writer.println(this.toString());
		writer.println();
		writer.close();
	}
	
	public void deposit(double amount) {
		balance = balance + amount;
		Transaction t = new Transaction(new Date(), "Deposit", amount, balance);
		writeTransaction(t);
		System.out.println(amount + " deposited successfully");
	}
	
	public void withdraw(double amount) {
		balance = balance - amount;
		Transaction t = new Transaction(new Date(), "Withdrawal", amount, balance);
		writeTransaction(t);
		System.out.println(amount + " withdrawn successfully");
	}
	
	public void writeTransaction(Transaction t) {
		try {
			writer = new PrintWriter(new FileWriter(transactions, true));
			writer.println(t.toString());
			writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	public String toString() {
		return ("Account owner: " + name + "\nAccount number: " + accountNumber + 
				"\nCurrent balance: " + balance + "\nAnnual interest: " + annualInterest + "%");
	}
	
	public void displayTransactions() throws IOException {
		Scanner reader = new Scanner(transactions);
		while(reader.hasNext()) {
			if(reader.hasNext("Date:")) {
				break;
			}
			reader.nextLine();
		}
		while(reader.hasNext()) {
			System.out.println(reader.nextLine());
		}
		reader.close();
		
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
