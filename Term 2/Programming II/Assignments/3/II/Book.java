package hw3;

public class Book {
	
	private String title;
	private String author;
	private double price;
	private int year;
	
	public Book(String title, String author, double price, int year) {
		this.title = title;
		this.author = author;
		this.price = price;
		this.year = year;
	}
	
	public String getTitle() {
		return title;
	}
	
	public String getAuthor() {
		return author;
	}
	
	public double getPrice() {
		return price;
	}
	
	public int getYear() {
		return year;
	}
	
	public void setTitle(String title) {
		this.title = title;
	}
	
	public void setAuthor(String author) {
		this.author = author;
	}
	
	public void setPrice(double price) {
		this.price = price;
	}
	
	public void setYear(int year) {
		this.year = year;
	}

}
