/* \author Aaron Brown */
// Quiz on implementing kd tree


// Structure to represent node of kd tree
struct Node
{
	std::vector<float> point;
	int id;
	Node* left;
	Node* right;

	Node(std::vector<float> arr, int setId)
	:	point(arr), id(setId), left(NULL), right(NULL)
	{}
};

struct KdTree
{
	Node* root;

	KdTree()
	: root(NULL)
	{}

	void insertHelper (Node** node, uint depth, std::vector<float> point, int id)
	{
		// If root is NULL, we fill it with the new point and its id
		if (*node == NULL)
			*node = new Node(point, id);
		else
		{
			// Check if depth is odd or even to know if we compare X or Y
			uint cd = depth%2;

			// If cd is 0, the depth is even, so X to be compared (its index in the vector point is also 0)
			// If cd is 1, the depth is odd, so Y to be compared (its index in the vector point is also 1)
			// The function will stop only when we find a NULL node in which we store the current point
			// As long as the node has already a 'branch', we need to check that branch to see if we allocate
			// the point after it in the next node
			if ( point[cd] < ((*node)->point[cd]) )
				insertHelper(&((*node)->left), depth+1, point, id);
			else
				insertHelper(&((*node)->right), depth+1, point, id);
		}
	}

	void insert(std::vector<float> point, int id)
	{
		// We send the root (even if at beginning it is NULL), a depth = 0, the received point and its id
		insertHelper(&root, 0, point, id);
	}

	void searchHelper(Node **node,int depth,std::vector<int> &ids,std::vector<float> target, float distanceTol)
	{
		if((*node)!=NULL){
			if (((target[0]-distanceTol)<= (*node)->point[0]) && ((target[0]+distanceTol) >= (*node)->point[0]) && 
			((target[1]+distanceTol) >= (*node)->point[1]) && ((target[1]-distanceTol)<= (*node)->point[1]) 
			/*&& ((target[2]+distanceTol) >= (*node)->point[2]) && ((target[2]-distanceTol)<= (*node)->point[2])*/  ){
				
				float distance = sqrt(((*node)->point[0]-target[0])*((*node)->point[0]-target[0]) + ((*node)->point[1]-target[1])*((*node)->point[1]-target[1]));
				if(distance<= distanceTol)
					ids.push_back((*node)->id);
			}
			uint cd= depth%2;
			//check across boundary
			if ((target[cd]-distanceTol)<=((*node)->point[cd]))	 // Point at left or below the node
				searchHelper(&((*node)->left),depth+1,ids, target,  distanceTol);
			if((target[cd]+distanceTol)>=((*node)->point[cd]))	// Point at right or above the node
				searchHelper(&((*node)->right),depth+1,ids, target,  distanceTol);
			}
	}

	// return a list of point ids in the tree that are within distance of target
	std::vector<int> search(std::vector<float> target, float distanceTol)
	{
		std::vector<int> ids;
		searchHelper(&root,0,ids,target, distanceTol);	

		return ids;
	}
};