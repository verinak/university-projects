
public class Employee {
	private int number;
	private float salary;
	
	//constructor that sets all values to 0
	public Employee() {
		number = 0;
		salary = 0;
	}
	
	//constructor that takes an int as the number attribute
	public Employee(int number) {
		this.number = number;
	}
	
	//constructor that takes a float as the salary attribute
	public Employee(float salary) {
		this.salary = salary;
	}
	
	//contructor that takes number and salary
	public Employee(int number, float salary) {
		this.number = number;
		this.salary = salary;
	}
	
	//mutator and accessor methods
	public int getNumber() {
		return number;
	}
	
	public float getSalary() {
		return salary;
	}
	
	public void setNumber(int number) {
		this.number = number;
	}
	
	public void setSalary(float salary) {
		this.salary = salary;
	}
	
	//method to display attributes to user
	public void displayEmployeeInfo() {
		System.out.println("Number: " + number + "\nSalary: " + salary);
	}
	
}
