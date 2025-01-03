/*
 * Displays the ImGui test window.
 *
 * This is free and unencumbered software released into the public domain. 
 */
#define IMGUI_DEFINE_MATH_OPERATORS


#include "bigg.hpp"
#include "value.h"
#include <imnodes.h>
#include "imgui.h"
#include "node_editor.h"
#include "context.h"
#include "imgui_canvas.h"


class NNGarden : public bigg::Application
{
	void onReset() {
		bgfx::setViewClear( 0, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0x303841ff, 1.0f, 0 );
		bgfx::setViewRect( 0, 0, 0, uint16_t( getWidth() ), uint16_t( getHeight() ) );
	}
	Context s_context;

	int  frame_count = 0;
	bool graph_open = true;
	bool data_open = true;
	bool training_menu_open = true;
	bool functions_open = true;
	bool demo_open = false;
	bool perf_open = false;

	void show_training_menu(bool* open) {
		s_context.show_training_menu(open);
	}

	void show_data(bool* open) {
		s_context.show_data(open);
	}

	void show_function_list(bool* open) {
		s_context.show_function_list(open);
	}

	void show_graph_editor(bool* open) {
		s_context.show(open);
	}

	void save(const char* filename) {
		s_context.save(filename);
	}

	void load(const char* filename) {
		s_context.load(filename);
	}

	void initialize(int _argc, char** _argv) {
		ImNodes::CreateContext(0);
		int width = 0;
		int height = 0;
		int comp = 0;
		
		s_context.main_graph.initialise();
		load("graph.json");
	}

	void update( float dt ) {
		bgfx::touch( 0 );
		if (frame_count == 0) {
		}

		frame_count++;

		ImGui::BeginMainMenuBar();

		if (ImGui::BeginMenu("File"))
		{

			if (ImGui::MenuItem("Save")) {
				save("graph.json");
			}

			if (ImGui::MenuItem("Load")) {
				load("graph.json");
			}

			ImGui::EndMenu();
		}

		if (ImGui::BeginMenu("Window"))
		{
			ImGui::MenuItem("Graph Window", "", &graph_open);
			ImGui::MenuItem("Functions", "", &functions_open);
			ImGui::MenuItem("Training Menu", "", &training_menu_open);
			ImGui::MenuItem("Data Menu", "", &data_open);
			ImGui::MenuItem("Demo Window", "", &demo_open);
			ImGui::MenuItem("Performance", "", &perf_open);

			//ImGui::MenuItem("Graph");
			//ImGui::EndMenu("Control");
			ImGui::EndMenu();
		}

		ImGui::EndMainMenuBar();

		if (ImGui::BeginPopup("SavePopup", ImGuiWindowFlags_MenuBar)) {
			ImGui::Text("Save");
			ImGui::EndPopup();
		}

		ImGui::DockSpaceOverViewport();

		//example::NodeEditorShow(dt, &graph_open);

		show_data(&data_open);
		show_training_menu(&training_menu_open);
		show_function_list(&functions_open);
		show_graph_editor(&graph_open);
		

		// Rendering
		//ImGui::Render();
		if (perf_open) {
			ImGui::Begin("Performance", &perf_open);
			ImGui::Text("%.1f", ImGui::GetIO().Framerate);
			ImGui::End();
		}
		if (demo_open)
			ImGui::ShowDemoWindow(&demo_open);
	}
public:
	NNGarden()
		: bigg::Application( "MLGarden" ) {
		printf("NNGarden constructor\n");

	}
};

NNGarden app;

int main( int argc, char** argv ) {
	return app.run( argc, argv );
}