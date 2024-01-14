package hw7;

public class triangleMain {

	public static void main(String[] args) {
		try {
			Triangle t1 = new Triangle(3,4,5);
			Triangle t2 = new Triangle(1,4,8);	//illegal sides
			Triangle t3 = new Triangle(6,14,10);
		}
		catch(IllegalTriangleException e) {
			System.out.println(e);
		}
		System.out.println("Triangles created: " + Triangle.getTrianglesCreated());

	}

}
