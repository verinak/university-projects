package hw3;

public class Library {
	
	private static int maxBooks = 100; 
	private Book[] books;
	private int count;
	
	public Library(Book[] books) {
		this.books = new Book[maxBooks];
		count = books.length;
		for(int i = 0; i < books.length; i++) {
			this.books[i] = books[i];
		}
	}
	
	public void addBook(Book book) {
		books[count] = book;
		count++;
	}
	
	public Book getBook(int i) {
		return books[i];
	}
	
	public int getCount() {
		return count;
	}
	
	public void printBookList() {
		for(int i = 0; i < count; i++) {
			System.out.println(books[i].getTitle() + " - " + books[i].getAuthor());
		}
	}

}
