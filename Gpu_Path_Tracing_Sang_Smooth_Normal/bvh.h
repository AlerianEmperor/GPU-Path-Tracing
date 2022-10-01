#ifndef _NEW_BVH_H_
#define _NEW_BVH_H_

#include "BBox.h"

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>

#define ct 4
#define ci 1

using namespace std;

float max_coord_after_scale = 1.2f;

struct node
{
	BBox box;

	node *nodes[2];// = { NULL };

	unsigned start : 25;
	unsigned range : 4;
	unsigned axis : 2;
	unsigned leaf : 1;

	node() {}
	node(int s, int r) : start(s), range(r) {}
};

struct FlatNode
{
	BBox box;

	unsigned start;
	int16_t range;
	int8_t axis;
	bool leaf;

	//unsigned int start : 25;
	//unsigned int range : 3;
	//unsigned int axis  : 2;
	//unsigned int leaf  : 1;
};

struct Triangle
{
	vec3 p0;
	vec3 e1;
	vec3 e2;
	vec3 n;

	Triangle() {}
	Triangle(vec3 p0_, vec3 e1_, vec3 e2_, vec3 n_) : p0(p0_), e1(e1_), e2(e2_), n(n_) {}
};

struct SimpleTriangle
{
	vec3 c;
	int i;
	BBox b;
	SimpleTriangle() {}
	SimpleTriangle(vec3 c_, int i_) : c(c_), i(i_) {}
};

struct Vertex
{
	vec3 p;
	vec3 n;

	Vertex() {}
	Vertex(vec3 p_, vec3 n_) : p(p_), n(n_) {}
	Vertex(float px, float py, float pz, float nx, float ny, float nz) : p(px, py, pz), n(nx, ny, nz) {}
};


//build vector
vector<FlatNode> flat_nodes;
vector<Vertex> vts;
vector<Triangle> trs;
vector<int> triangle_index;

int num_vertices;
int num_triangles;
int num_nodes;

//cuda vector
//Vertex* vertices;
//Triangle* triangles;



//Read Object

void load_ply(string& filename)
{
	ifstream ifs(filename);

	string line;

	int num_vertex = 0;
	int num_triangle = 0;

	vec3 min_vec(INT_MAX, INT_MAX, INT_MAX);
	vec3 max_vec(INT_MIN, INT_MIN, INT_MIN);

	while (getline(ifs, line))
	{
		if (line.substr(0, 14) == "element vertex")
		{
			istringstream iss(line);

			string word1;

			iss >> word1;
			iss >> word1;
			iss >> num_vertex;

			num_vertices = num_vertex;

			//vertices = (Vertex*)malloc(sizeof(Vertex) * num_vertex);
			//p_current_vertex = vertices;
		}
		else if (line.substr(0, 12) == "element face")
		{
			istringstream iss(line);

			string word1;

			iss >> word1;
			iss >> word1;
			iss >> num_triangle;

			num_triangles = num_triangle;

			//triangles = (Triangle*)malloc(sizeof(Triangle) * num_triangle);
			//p_current_triangle = triangles;

			//trs.resize()
		}
		else if (line.substr(0, 10) == "end_header")
			break;
	}

	int num_v = num_vertex;
	int num_tr = num_triangle;


	//cout << num_vertex << " " << num_triangle << "\n";

	while (getline(ifs, line))
	{
		if (num_v)
		{
			--num_v;

			istringstream iss(line);

			float x, y, z;

			iss >> x >> y >> z;

			//cout << x << " " << y << " " << z << "\n";
			//p_current_vertex->p = vec3(x, y, z);
			//p_current_vertex->y = y;
			//p_current_vertex->z = z;

			//p_current_vertex++;
			//vertices.push_back(vec3(x, y, z));

			vec3 v(x, y, z);

			vts.push_back({ v, vec3(0, 0, 0) });

			min_vec = min3(min_vec, v);
			max_vec = max3(max_vec, v);

			//cout << x << " " << y << " " << z << "\n";
		}
		else
		{
			istringstream iss(line);

			int num, x, y, z;

			iss >> num >> x >> y >> z;

			//cout << x << " " << y << " " << z << "\n";

			triangle_index.push_back(x);
			triangle_index.push_back(y);
			triangle_index.push_back(z);

			vec3 v1 = vts[x].p;
			vec3 v2 = vts[y].p;
			vec3 v3 = vts[z].p;

			vec3 p0 = v1;
			vec3 e1 = v2 - v1;
			vec3 e2 = v3 - v1;
			vec3 n = (e1.cross(e2));// .norm();

			//p_current_triangle->p0 = p0;
			//p_current_triangle->e1 = e1;
			//p_current_triangle->e2 = e2;
			//p_current_triangle->n = n;

			//p_current_triangle++;


			//smooth normal
			vts[x].n += n;
			vts[y].n += n;
			vts[z].n += n;

			//triangles.push_back({ p0, e1, e2 });

			trs.push_back({ p0, e1, e2, n.norm() });
		}
	}

	vec3 center = (min_vec + max_vec) * 0.5f;

	//min_vec -= center;
	//max_vec -= center;

	float maxi = 0;
	maxi = maxf(maxi, (float)fabs(min_vec.x));
	maxi = maxf(maxi, (float)fabs(min_vec.y));
	maxi = maxf(maxi, (float)fabs(min_vec.z));
	maxi = maxf(maxi, (float)fabs(max_vec.x));
	maxi = maxf(maxi, (float)fabs(max_vec.y));
	maxi = maxf(maxi, (float)fabs(max_vec.z));

	//smooth normal
	for (int i = 0; i < num_vertices; ++i)
	{
		vts[i].p -= center;
		vts[i].p *= (max_coord_after_scale / maxi);
		vts[i].n.normalize();
	}

	for (int i = 0; i < num_triangles; ++i)
	{
		int x = triangle_index[3 * i];
		int y = triangle_index[3 * i + 1];
		int z = triangle_index[3 * i + 2];

		vec3 v1 = vts[x].p;
		vec3 v2 = vts[y].p;
		vec3 v3 = vts[z].p;

		vec3 p0 = v1;
		vec3 e1 = (v2 - v1);// .norm();
		vec3 e2 = (v3 - v1);// .norm();

		vec3 n = (e1.cross(e2)).norm();

		trs.push_back({ p0, e1, e2, n });
	}


	//cout << "Static: \n";
	
	//cout << trs.size() << "\n";
	//cout << triangle_index.size() << "\n";
	//cout << vts.size() << "\n";

}


void sah(node*& n, const int& start, const int& range, int& leaf, vector<BBox>& boxes, vector<SimpleTriangle>& simp)
{
	n = new node(start, range);//range
	for (auto i = start; i < start + range; ++i)//<start + range
	{
		int ind = simp[i].i;
		simp[i].b = boxes[ind];
		n->box.expand(boxes[ind]);
	}
	if (range < leaf)
	{
		n->leaf = 1;
		n->axis = n->box.maxDim();
		int axis = n->axis;
		sort(simp.begin() + start, simp.begin() + start + range, [axis](const SimpleTriangle& s1, const SimpleTriangle& s2)
		{
			return s1.c[axis] < s2.c[axis];
		});
		return;
	}
	else
	{
		n->leaf = 0;
		n->range = 0;
		int best_split = 0, best_axis = -1;

		float best_cost = ci * range;
		float area = n->box.area();
		vec3 vmin = n->box.bbox[0], vmax = n->box.bbox[1];
		vec3 extend(vmax - vmin);

		for (int a = 0; a < 3; ++a)
		{
			sort(simp.begin() + start, simp.begin() + start + range, [a](const SimpleTriangle& s1, const SimpleTriangle& s2)
			{
				return s1.c[a] <= s2.c[a];
			});

			//float min_box = n->box.bbox[0][a], length = n->box.bbox[1][a] - min_box;
			float length = n->box.bbox[1][a] - n->box.bbox[0][a];
			//if (length < 0.000001f)
			//	continue;

			vector<BBox> right_boxes;
			right_boxes.resize(range);

			BBox left = simp[start + 0].b;

			right_boxes[range - 1] = simp[start + range - 1].b;

			for (int j = range - 2; j >= 0; --j)
			{
				right_boxes[j] = right_boxes[j + 1].expand_box(simp[start + j].b);
			}

			float extend = length / range;

			int count_left = 1;
			int count_right = range - 1;
			float inv_a = 1.0f / area;
			for (int i = 0; i < range - 1; ++i)
			{
				float left_area = left.area();
				float right_area = right_boxes[i + 1].area();

				BBox right = right_boxes[i + 1];

				float cost = ct + ci * (left_area * count_left + right_area * count_right) * inv_a;

				if (cost < best_cost)
				{
					best_cost = cost;
					best_axis = a;
					best_split = count_left;
				}
				++count_left;
				--count_right;
				left.expand(simp[start + i + 1].b);
			}
		}
		if (best_cost == ci * range)
		{
			n->leaf = 1;
			n->range = range;
			n->axis = n->box.maxDim();
			int axis = n->axis;
			sort(simp.begin() + start, simp.begin() + start + range, [axis](const SimpleTriangle& s1, const SimpleTriangle& s2)
			{
				return s1.c[axis] < s2.c[axis];
			});
			return;
		}
		if (best_split == 0 || best_split == range)//turn into leaf	//42.232s 42.393s
		{
			best_split = range / 2;
		}
		n->axis = best_axis;

		sort(simp.begin() + start, simp.begin() + start + range, [best_axis](const SimpleTriangle& s1, const SimpleTriangle& s2)
		{
			return s1.c[best_axis] < s2.c[best_axis];
		});

		sah(n->nodes[0], start, best_split, leaf, boxes, simp);
		//sah(n->left, start + best_split, range - best_split, boxes, simp);
		sah(n->nodes[1], start + best_split, range - best_split, leaf, boxes, simp);
	}
}

void flat_bvh(node*& n, vector<FlatNode>& flat_nodes)
{
	queue<node*> queue_node;

	queue_node.emplace(n);
	int current_index = 0;
	int left_index = 0;

	while (!queue_node.empty())
	{
		node* front = queue_node.front();

		queue_node.pop();

		if (front->leaf)
		{
			FlatNode flatnode;

			flatnode.box = front->box;
			flatnode.start = front->start;
			flatnode.range = front->range;
			flatnode.axis = front->axis;
			flatnode.leaf = true;

			//flat_nodes[current_index++] = flatnode;
			flat_nodes.emplace_back(flatnode);
			//++left_index;
		}
		else
		{
			FlatNode flatnode;

			flatnode.box = front->box;
			flatnode.start = ++left_index;
			flatnode.range = 0;
			flatnode.axis = front->axis;
			flatnode.leaf = false;

			//flat_nodes[current_index++] = flatnode;

			flat_nodes.emplace_back(flatnode);

			queue_node.emplace(front->nodes[0]);
			queue_node.emplace(front->nodes[1]);

			++left_index;
		}
	}


}

void inordere_bvh(node*& root)
{
	//if (!root)
	//	return;

	//cout << root->start << " " << root->range << "\n";

	//inordere_bvh(root->nodes[0]);
	//inordere_bvh(root->nodes[1]);

	for (int i = 0; i < num_nodes; ++i)
		cout << flat_nodes[i].start << " " << flat_nodes[i].range<< " " << flat_nodes[i].leaf << "\n";
}

void build_bvh(node*& root, const int& l)
{
	int leaf = l;
	int s = trs.size();

	vector<BBox> boxes;
	vector<SimpleTriangle> simp;
	boxes.resize(s);
	simp.resize(s);

	for (int i = 0; i < s; ++i)
	{
		vec3 p0 = trs[i].p0;

		boxes[i].expand(p0);
		boxes[i].expand(p0 + trs[i].e1);
		boxes[i].expand(p0 + trs[i].e2);
		simp[i] = { boxes[i].c(), i };
	}
	//int depth = 0;

	//node* root;
	sah(root, 0, s, leaf, boxes, simp);

	
	//inordere_bvh(root);

	vector<FlatNode> flat;
	flat_bvh(root, flat);

	flat_nodes = flat;

	

	vector<Triangle> new_trs;

	new_trs.resize(s);

	for (int i = 0; i < s; ++i)
		new_trs[i] = trs[simp[i].i];

	trs = new_trs;


	num_nodes = flat_nodes.size();

	//cout << "flat: " << flat_nodes.size() << "\n";

	//inordere_bvh(root);

	/*for (int i = 0; i < num_triangles; ++i)
	{
		cout << trs[i].p0.x << " " << trs[i].p0.y << " " << trs[i].p0.z << "\n";
		cout << trs[i].e1.x << " " << trs[i].e1.y << " " << trs[i].e1.z << "\n";
		cout << trs[i].e2.x << " " << trs[i].e2.y << " " << trs[i].e2.z << "\n";
		cout << trs[i].n.x << " " << trs[i].n.y << " " << trs[i].n.z << "\n\n";

	}*/

	vector<SimpleTriangle>().swap(simp);


	vector<Triangle>().swap(new_trs);
}

/*
float* triangles_value;
int* triangle_indices;
float* bvh_boxes;
int* bvh_indices;
float* vertex_normal;
*/
//copy from vector to cuda memory

/*
float* cuda_triangles_value;
int* cuda_triangle_indices;
float* cuda_bvh_boxes;
int* cuda_bvh_indices;
float* cuda_vertex_normal;

//step 1
//copy from host to temporary memory

//step 2
//copy from temporary to device
//Note: the function to copy temporary memory to device is located in render.cu

float* triangles_value;// = new float[16 * num_triangles];
int* triangle_indices;// = new int[4 * num_triangles];

					  // = flat_nodes.size();

float* bvh_boxes;// = new float[6 * num_nodes];
int* bvh_indices;// = new int[4 * num_nodes];

float* vertex_normal;// = new float[4 * num_vertices];

					 //step 1
void copy_host_to_temporary_memory()
{


	//float* triangles_value;
	//int* triangle_indices;
	//float* bvh_boxes;
	//int* bvh_indices;
	//float* vertex_normal;

	triangles_value = new float[16 * num_triangles];
	triangle_indices = new int[4 * num_triangles];

	//num_nodes = flat_nodes.size();

	bvh_boxes = new float[6 * num_nodes];
	bvh_indices = new int[4 * num_nodes];

	vertex_normal = new float[4 * num_vertices];

	for (int i = 0; i < num_triangles; ++i)
	{
		triangles_value[16 * i] = trs[i].p0.x;
		triangles_value[16 * i + 1] = trs[i].p0.y;
		triangles_value[16 * i + 2] = trs[i].p0.z;
		triangles_value[16 * i + 3] = 0.0f;

		triangles_value[16 * i + 4] = trs[i].e1.x;
		triangles_value[16 * i + 5] = trs[i].e1.y;
		triangles_value[16 * i + 6] = trs[i].e1.z;
		triangles_value[16 * i + 7] = 0.0f;

		triangles_value[16 * i + 8] = trs[i].e2.x;
		triangles_value[16 * i + 9] = trs[i].e2.y;
		triangles_value[16 * i + 10] = trs[i].e2.z;
		triangles_value[16 * i + 11] = 0.0f;

		triangles_value[16 * i + 12] = trs[i].n.x;
		triangles_value[16 * i + 13] = trs[i].n.y;
		triangles_value[16 * i + 14] = trs[i].n.z;
		triangles_value[16 * i + 15] = 0.0f;

		triangle_indices[4 * i] = triangle_index[3 * i];
		triangle_indices[4 * i + 1] = triangle_index[3 * i + 1];
		triangle_indices[4 * i + 2] = triangle_index[3 * i + 2];
		triangle_indices[4 * i + 3] = 0;//material index, currently not use
	}

	for (int i = 0; i < num_nodes; ++i)
	{
		bvh_boxes[6 * i] = flat_nodes[i].box.bbox[0].x;
		bvh_boxes[6 * i + 1] = flat_nodes[i].box.bbox[1].x;

		bvh_boxes[6 * i + 2] = flat_nodes[i].box.bbox[0].y;
		bvh_boxes[6 * i + 3] = flat_nodes[i].box.bbox[1].y;

		bvh_boxes[6 * i + 4] = flat_nodes[i].box.bbox[0].z;
		bvh_boxes[6 * i + 5] = flat_nodes[i].box.bbox[1].z;

		bvh_indices[4 * i] = flat_nodes[i].start;
		bvh_indices[4 * i + 1] = flat_nodes[i].range;
		bvh_indices[4 * i + 2] = flat_nodes[i].axis;
		bvh_indices[4 * i + 3] = flat_nodes[i].leaf;
	}



	for (int i = 0; i < num_vertices; ++i)
	{
		vertex_normal[4 * i] = vts[i].n.x;
		vertex_normal[4 * i + 1] = vts[i].n.y;
		vertex_normal[4 * i + 2] = vts[i].n.z;
		vertex_normal[4 * i + 3] = 0.0f;//pad
	}

	
	//float* cuda_triangles_value;
	//int* cuda_triangle_indices;
	//float* cuda_bvh_boxes;
	//int* cuda_bvh_indices;
	//float* cuda_vertex_normal;
	

	cudaMalloc((void**)cuda_triangles_value, num_triangles * 16 * sizeof(float));
	cudaMemcpy(cuda_triangles_value, triangles_value, num_triangles * 16 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)cuda_triangle_indices, num_triangles * 4 * sizeof(int));
	cudaMemcpy(cuda_triangle_indices, triangle_indices, num_triangles * 4 * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)cuda_bvh_boxes, num_nodes * 6 * sizeof(float));
	cudaMemcpy(cuda_bvh_boxes, bvh_boxes, num_nodes * 6 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)cuda_bvh_indices, num_nodes * 4 * sizeof(int));
	cudaMemcpy(cuda_bvh_indices, bvh_indices, num_nodes * 4 * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)cuda_vertex_normal, num_vertices * 4 * sizeof(float));
	cudaMemcpy(cuda_vertex_normal, vertex_normal, num_vertices * 4 * sizeof(float), cudaMemcpyHostToDevice);
}
*/


void Read_Scene(node*& root, string& file_name)
{
	load_ply(file_name);

	int l = 2;

	build_bvh(root, l);

	//inordere_bvh(root);

	num_nodes = flat_nodes.size();

	//cout << "flat node: " << num_nodes << "\n";


	//inordere_bvh(root);

	//cout <<  "num_triangles" << " " << num_triangles<<"\n";

	//copy_host_to_temporary_memory();
	//copy_temporary_to_device(triangles_value, triangle_indices, bvh_boxes, bvh_indices,
	//	vertex_normal, num_triangles, num_vertices, num_nodes);

	//copy_temporary_to_device(float* triangles_value, int* triangle_indices, float* bvh_boxes, int* bvh_indices,
	//	float* vertex_normal, int num_triangles, int num_vertices, int num_nodess)
}


#endif // !_BVH_H_

