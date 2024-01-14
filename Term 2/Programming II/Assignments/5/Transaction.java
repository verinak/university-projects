package hw5;

import java.util.Date;

public class Transaction {
	
	private Date date;
	private char type;
	private double amount;
	private double balanceAfter;
	
	
	public Transaction(Date date, char type, double amount, double balanceAfter) {
		this.date = date;
		this.type = type;
		this.amount = amount;
		this.balanceAfter = balanceAfter;
	}
	
	public String toString() {
		return ("Date: " + date + "\nType: " + type + "\nAmount: " + amount + "\nNew Balance: "
	+ balanceAfter + "\n------------------------------------\n");
	}


	public Date getDate() {
		return date;
	}


	public void setDate(Date date) {
		this.date = date;
	}


	public char getType() {
		return type;
	}


	public void setType(char type) {
		this.type = type;
	}


	public double getAmount() {
		return amount;
	}


	public void setAmount(double amount) {
		this.amount = amount;
	}


	public double getBalanceAfter() {
		return balanceAfter;
	}


	public void setBalanceAfter(double balanceAfter) {
		this.balanceAfter = balanceAfter;
	}

}
