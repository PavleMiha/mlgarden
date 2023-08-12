/*
 * Displays the ImGui test window.
 *
 * This is free and unencumbered software released into the public domain. 
 */

#include "bigg.hpp"
#include "value.h"
#include "imgui.h"
#include <imnodes.h>
#include "node_editor.h"

class NNGarden : public bigg::Application
{
	void initialize(int _argc, char** _argv) {
		ImNodes::CreateContext();
		example::NodeEditorInitialize();
	}
	void onReset() {
		bgfx::setViewClear( 0, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0x303841ff, 1.0f, 0 );
		bgfx::setViewRect( 0, 0, 0, uint16_t( getWidth() ), uint16_t( getHeight() ) );
	}

	void update( float dt ) {
		bgfx::touch( 0 );

		//ImGui::NewFrame();

		example::NodeEditorShow(dt);

		// Rendering
		//ImGui::Render();
		ImGui::ShowDemoWindow();
	}
public:
	NNGarden()
		: bigg::Application( "NNGarden" ) {}
};

int main( int argc, char** argv ) {
	NNGarden app;
	return app.run( argc, argv );
}