package project;

import java.util.Date;

public class Transaction {
	
	private Date date;
	private String type;
	private double amount;
	private double balanceAfter;
	private String note;
	
	public Transaction(Date date, String type, double amount, double balanceAfter) {
		this.date = date;
		this.type = type;
		this.amount = amount;
		this.balanceAfter = balanceAfter;
	}
	
	public Transaction(Date date, String type, double amount, double balanceAfter, String note) {
		this(date, type, amount, balanceAfter);
		this.note = note;
	}
	
	public String toString() {
		return ("Date: " + date + "\nType: " + type + "\nAmount: " + amount + "\nNew Balance: "
	+ balanceAfter + "\nAdditional notes: " + note + "\n------------------------------------\n");
	}


	public Date getDate() {
		return date;
	}


	public void setDate(Date date) {
		this.date = date;
	}


	public String getType() {
		return type;
	}


	public void setType(String type) {
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
	
	public String getNote() {
		return note;
	}
	
	public void setNote(String note) {
		this.note = note;	
	}

}
