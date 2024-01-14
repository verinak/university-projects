package hw3;

public class MainLibrary {

	public static void main(String[] args) {
		
		Book[] tbr = new Book[5];
		tbr[0] = new Book("Priory of the Orange Tree", "Samantha Shannon", 17.99, 2019);
		tbr[1] = new Book("The Atlas Six", "Olivie Blake", 18.99, 2020);
		tbr[2] = new Book("The Final Empire", "Brandon Sanderson", 8.99, 2006);
		tbr[3] = new Book("The Guilded Wolves", "Roshani Chokshi", 11.99, 2018);
		tbr[4] = new Book("Lore", "Alexandra Bracken", 12.99, 2021);
				
		Library myLibrary = new Library(tbr);
		
		Book b1 = new Book("The Bone Ships", "RJ Barker", 9.99, 2019);
		myLibrary.addBook(b1);
		
		System.out.println("Number of books: " + myLibrary.getCount());
		System.out.println();
		myLibrary.printBookList();
		

	}

}
